const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');
const OpenAI = require('openai');

/**
 * --------------------------------------------------------------------------
 * 고정 설정
 * --------------------------------------------------------------------------
 */

// 현재 Firebase knowledge_chunks의 embedding과 동일하게 유지
const EMBEDDING_MODEL = 'gemini-embedding-001';
const EMBEDDING_DIMENSIONS = 768;

// 실제 답변 모델
const MODELS = {
  gemini: 'gemini-3.1-flash-lite',
  gpt: 'gpt-5.6-luna',
};

// 검색 설정
const DEFAULT_TOP_K = 5;
const DEFAULT_DISTANCE_THRESHOLD = 0.70;

// 서버가 관리하는 시스템 프롬프트
const SYSTEM_PROMPT = `
너는 "사주그랩 기술 백서"를 근거로 답변하는 RAG(검색증강생성) 어시스턴트다.

반드시 다음 규칙을 지켜라.

1. 제공된 [백서 근거]를 가장 중요한 사실 근거로 사용한다.
2. 백서 근거에 없는 내용을 사실처럼 만들어내지 않는다.
3. 근거가 충분하지 않으면 "제공된 백서에서 확인할 수 없습니다."라고 명확하게 답한다.
4. 숫자, 공식, 가중치, 알고리즘, 구조에 관한 내용은 백서의 의미를 임의로 변경하지 않는다.
5. 백서에 없는 일반 지식을 덧붙일 경우 백서의 내용과 구분한다.
6. 질문에 직접 답하고 불필요하게 장황하게 설명하지 않는다.
`.trim();

/**
 * --------------------------------------------------------------------------
 * Firebase 초기화
 * --------------------------------------------------------------------------
 */

function getDb() {
  if (!getApps().length) {
    const rawEnv = process.env.FIREBASE_SERVICE_ACCOUNT;

    if (!rawEnv) {
      throw new Error(
        "FIREBASE_SERVICE_ACCOUNT 환경변수가 설정되지 않았습니다."
      );
    }

    let serviceAccount;

    try {
      serviceAccount = JSON.parse(rawEnv);
    } catch (error) {
      throw new Error(
        `FIREBASE_SERVICE_ACCOUNT JSON 파싱 실패: ${error.message}`
      );
    }

    if (!serviceAccount.project_id) {
      throw new Error(
        'Firebase service account에 project_id가 없습니다.'
      );
    }

    if (!serviceAccount.client_email) {
      throw new Error(
        'Firebase service account에 client_email이 없습니다.'
      );
    }

    if (!serviceAccount.private_key) {
      throw new Error(
        'Firebase service account에 private_key가 없습니다.'
      );
    }

    serviceAccount.private_key =
      serviceAccount.private_key.replace(/\\n/g, '\n');

    initializeApp({
      credential: cert(serviceAccount),
    });
  }

  return getFirestore();
}

/**
 * --------------------------------------------------------------------------
 * 입력 검증
 * --------------------------------------------------------------------------
 */

function parseBody(req) {
  if (!req.body) {
    return {};
  }

  if (typeof req.body === 'object') {
    return req.body;
  }

  try {
    return JSON.parse(req.body);
  } catch {
    throw new Error(
      '요청 body가 유효한 JSON이 아닙니다.'
    );
  }
}

function normalizeUserQuery(value) {
  if (typeof value !== 'string') {
    throw new Error('message는 문자열이어야 합니다.');
  }

  const query = value.trim();

  if (!query) {
    throw new Error('질문을 입력해주세요.');
  }

  if (query.length > 10000) {
    throw new Error(
      '질문이 너무 깁니다. 10,000자 이하로 입력해주세요.'
    );
  }

  return query;
}

function normalizeModel(value) {
  if (!value) {
    return MODELS.gemini;
  }

  if (
    value === 'gemini' ||
    value === MODELS.gemini
  ) {
    return MODELS.gemini;
  }

  if (
    value === 'gpt' ||
    value === MODELS.gpt
  ) {
    return MODELS.gpt;
  }

  throw new Error(
    `지원하지 않는 모델입니다: ${value}`
  );
}

function normalizeTopK(value) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return DEFAULT_TOP_K;
  }

  return Math.min(
    Math.max(Math.floor(parsed), 1),
    10
  );
}

function normalizeDistanceThreshold(value) {
  if (
    value === undefined ||
    value === null ||
    value === ''
  ) {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  return Math.min(
    Math.max(parsed, 0),
    2
  );
}

/**
 * --------------------------------------------------------------------------
 * 질문 임베딩
 * --------------------------------------------------------------------------
 *
 * 백서 embedding과 동일:
 * gemini-embedding-001 / 768차원
 *
 * --------------------------------------------------------------------------
 */

async function createQueryEmbedding(
  userQuery,
  geminiKey
) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/` +
    `models/${EMBEDDING_MODEL}:embedContent` +
    `?key=${encodeURIComponent(geminiKey)}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      content: {
        parts: [
          {
            text: userQuery,
          },
        ],
      },
      outputDimensionality: EMBEDDING_DIMENSIONS,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    const apiMessage =
      data?.error?.message ||
      JSON.stringify(data);

    throw new Error(
      `Gemini 임베딩 API 오류: ${apiMessage}`
    );
  }

  const values = data?.embedding?.values;

  if (!Array.isArray(values)) {
    throw new Error(
      `Gemini 임베딩 결과가 올바르지 않습니다: ${JSON.stringify(data)}`
    );
  }

  if (values.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `임베딩 차원이 맞지 않습니다. ` +
      `예상=${EMBEDDING_DIMENSIONS}, ` +
      `실제=${values.length}`
    );
  }

  return values;
}

/**
 * --------------------------------------------------------------------------
 * Firestore Vector Search
 * --------------------------------------------------------------------------
 */

async function retrieveKnowledge(
  db,
  queryVector,
  topK,
  distanceThreshold
) {
  const vectorQuery =
    db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: topK,
      distanceMeasure: 'COSINE',
      distanceResultField: 'vector_distance',
    });

  const snapshot =
    await vectorQuery.get();

  const searched =
    snapshot.docs
      .map((doc) => {
        const data = doc.data();

        return {
          id: doc.id,

          text:
            typeof data.text === 'string'
              ? data.text
              : '',

          source:
            data.source || null,

          chunk_index:
            typeof data.chunk_index === 'number'
              ? data.chunk_index
              : null,

          distance:
            typeof data.vector_distance === 'number'
              ? data.vector_distance
              : null,
        };
      })
      .filter((item) => item.text);

  // COSINE distance는 낮을수록 유사
  const matched =
    searched.filter((item) => {
      if (item.distance === null) {
        return true;
      }

      return (
        item.distance <=
        distanceThreshold
      );
    });

  return {
    searched,
    matched,
  };
}

/**
 * --------------------------------------------------------------------------
 * RAG Context
 * --------------------------------------------------------------------------
 */

function buildContext(chunks) {
  if (!chunks.length) {
    return (
      '[백서 근거]\n' +
      '관련 백서 청크를 찾지 못했습니다.'
    );
  }

  return chunks
    .map((chunk, index) => {
      const metadata = [
        `청크 ${index + 1}`,

        chunk.source
          ? `source=${chunk.source}`
          : null,

        chunk.chunk_index !== null
          ? `chunk_index=${chunk.chunk_index}`
          : null,

        chunk.distance !== null
          ? `cosine_distance=${chunk.distance.toFixed(6)}`
          : null,
      ]
        .filter(Boolean)
        .join(' | ');

      return (
        `[${metadata}]\n` +
        chunk.text
      );
    })
    .join(
      '\n\n--- 백서 청크 구분 ---\n\n'
    );
}

function buildUserPrompt(
  userQuery,
  retrievedChunks
) {
  return `
[백서 근거]

${buildContext(retrievedChunks)}

[사용자 질문]

${userQuery}

[답변 지침]

위 백서 근거를 바탕으로 질문에 답변하라.
근거가 충분하지 않으면
"제공된 백서에서 확인할 수 없습니다."라고 답하라.
`.trim();
}

/**
 * --------------------------------------------------------------------------
 * Gemini 답변
 * --------------------------------------------------------------------------
 */

async function generateGeminiAnswer(
  prompt,
  geminiKey
) {
  const ai =
    new GoogleGenAI({
      apiKey: geminiKey,
    });

  const response =
    await ai.models.generateContent({
      model: MODELS.gemini,

      contents: prompt,

      config: {
        systemInstruction:
          SYSTEM_PROMPT,
      },
    });

  const answer =
    response?.text?.trim();

  if (!answer) {
    throw new Error(
      'Gemini가 빈 답변을 반환했습니다.'
    );
  }

  return answer;
}

/**
 * --------------------------------------------------------------------------
 * GPT-5.6 Luna 답변
 * --------------------------------------------------------------------------
 *
 * temperature 사용 안 함.
 * GPT-5.6 reasoning.effort 사용.
 * --------------------------------------------------------------------------
 */

async function generateGptAnswer(
  prompt,
  openaiKey
) {
  const openai =
    new OpenAI({
      apiKey: openaiKey,
    });

  const response =
    await openai.responses.create({
      model: MODELS.gpt,

      instructions:
        SYSTEM_PROMPT,

      input:
        prompt,

      reasoning: {
        effort: 'medium',
      },
    });

  const answer =
    response?.output_text?.trim();

  if (!answer) {
    throw new Error(
      'GPT-5.6 Luna가 빈 답변을 반환했습니다.'
    );
  }

  return answer;
}

/**
 * --------------------------------------------------------------------------
 * Main API
 * --------------------------------------------------------------------------
 */

module.exports = async (
  req,
  res
) => {
  if (req.method !== 'POST') {
    res.setHeader(
      'Allow',
      'POST'
    );

    return res
      .status(405)
      .json({
        error:
          'Method not allowed',
      });
  }

  try {
    const body =
      parseBody(req);

    const userQuery =
      normalizeUserQuery(
        body.message
      );

    const selectedModel =
      normalizeModel(
        body.model
      );

    const topK =
      normalizeTopK(
        body.topK
      );

    const distanceThreshold =
      normalizeDistanceThreshold(
        body.distanceThreshold
      );

    const geminiKey =
      process.env.GEMINI_API_KEY;

    if (!geminiKey) {
      return res
        .status(500)
        .json({
          error:
            "GEMINI_API_KEY 환경변수가 설정되지 않았습니다.",
        });
    }

    if (
      selectedModel === MODELS.gpt &&
      !process.env.OPENAI_API_KEY
    ) {
      return res
        .status(500)
        .json({
          error:
            "OPENAI_API_KEY 환경변수가 설정되지 않았습니다.",
        });
    }

    const db =
      getDb();

    /**
     * 1. 질문 임베딩
     */
    const queryVector =
      await createQueryEmbedding(
        userQuery,
        geminiKey
      );

    /**
     * 2. Firestore vector search
     */
    const retrieval =
      await retrieveKnowledge(
        db,
        queryVector,
        topK,
        distanceThreshold
      );

    /**
     * 3. RAG prompt
     */
    const prompt =
      buildUserPrompt(
        userQuery,
        retrieval.matched
      );

    /**
     * 4. LLM
     */
    let answer;

    if (
      selectedModel === MODELS.gpt
    ) {
      answer =
        await generateGptAnswer(
          prompt,
          process.env.OPENAI_API_KEY
        );
    } else {
      answer =
        await generateGeminiAnswer(
          prompt,
          geminiKey
        );
    }

    /**
     * 5. 결과 반환
     */
    return res
      .status(200)
      .json({
        answer,

        model:
          selectedModel,

        retrieval: {
          topK,

          distanceThreshold,

          searchedCount:
            retrieval.searched.length,

          matchedCount:
            retrieval.matched.length,

          chunks:
            retrieval.matched.map(
              (chunk) => ({
                id:
                  chunk.id,

                source:
                  chunk.source,

                chunk_index:
                  chunk.chunk_index,

                distance:
                  chunk.distance,

                text:
                  chunk.text,
              })
            ),
        },
      });

  } catch (error) {
    console.error(
      'RAG API Error:',
      error
    );

    return res
      .status(500)
      .json({
        error:
          error?.message ||
          '서버 내부 오류가 발생했습니다.',
      });
  }
};
