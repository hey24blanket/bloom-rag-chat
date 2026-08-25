const express = require('express');
const path = require('path');

const {
  initializeApp,
  cert,
  getApps,
} = require('firebase-admin/app');

const {
  getFirestore,
  FieldValue,
} = require('firebase-admin/firestore');

const {
  GoogleGenAI,
} = require('@google/genai');

const OpenAI = require('openai');

const app = express();

const PORT =
  Number(process.env.PORT) || 3000;

/**
 * ============================================================================
 * 고정 설정
 * ============================================================================
 *
 * Firebase에 이미 저장된 백서 embedding:
 *
 *   gemini-embedding-001
 *   768 dimensions
 *
 * 절대 자동 탐색하지 않습니다.
 */

const EMBEDDING_MODEL =
  'gemini-embedding-001';

const EMBEDDING_DIMENSIONS =
  768;

const MODELS = {
  gemini:
    'gemini-3.1-flash-lite',

  gpt:
    'gpt-5.6-luna',
};

const DEFAULT_TOP_K = 5;

const DEFAULT_DISTANCE_THRESHOLD =
  0.70;

/**
 * ============================================================================
 * Firebase 초기화
 * ============================================================================
 */

function getDb() {
  if (!getApps().length) {
    const rawServiceAccount =
      process.env.FIREBASE_SERVICE_ACCOUNT;

    if (!rawServiceAccount) {
      throw new Error(
        'FIREBASE_SERVICE_ACCOUNT 환경변수가 설정되지 않았습니다.'
      );
    }

    let serviceAccount;

    try {
      serviceAccount =
        JSON.parse(
          rawServiceAccount
        );
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
      serviceAccount.private_key.replace(
        /\\n/g,
        '\n'
      );

    initializeApp({
      credential:
        cert(serviceAccount),
    });
  }

  return getFirestore();
}

/**
 * ============================================================================
 * 요청 body
 * ============================================================================
 */

function parseBody(req) {
  if (!req.body) {
    return {};
  }

  if (
    typeof req.body ===
    'object'
  ) {
    return req.body;
  }

  try {
    return JSON.parse(
      req.body
    );
  } catch {
    throw new Error(
      '요청 body가 유효한 JSON이 아닙니다.'
    );
  }
}

/**
 * ============================================================================
 * 입력 검증
 * ============================================================================
 */

function normalizeUserQuery(
  value
) {
  if (
    typeof value !==
    'string'
  ) {
    throw new Error(
      'message는 문자열이어야 합니다.'
    );
  }

  const query =
    value.trim();

  if (!query) {
    throw new Error(
      '질문을 입력해주세요.'
    );
  }

  if (
    query.length > 10000
  ) {
    throw new Error(
      '질문이 너무 깁니다. 10,000자 이하로 입력해주세요.'
    );
  }

  return query;
}

function normalizeSystemPrompt(
  value
) {
  if (
    typeof value !==
    'string'
  ) {
    throw new Error(
      'systemPrompt는 문자열이어야 합니다.'
    );
  }

  const prompt =
    value.trim();

  if (!prompt) {
    throw new Error(
      '시스템 프롬프트를 설정에서 입력해주세요.'
    );
  }

  if (
    prompt.length > 30000
  ) {
    throw new Error(
      '시스템 프롬프트가 너무 깁니다. 30,000자 이하로 입력해주세요.'
    );
  }

  return prompt;
}

function normalizeModel(
  value
) {
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

function normalizeTopK(
  value
) {
  const parsed =
    Number(value);

  if (
    !Number.isFinite(
      parsed
    )
  ) {
    return DEFAULT_TOP_K;
  }

  return Math.min(
    Math.max(
      Math.floor(parsed),
      1
    ),
    10
  );
}

function normalizeDistanceThreshold(
  value
) {
  if (
    value === undefined ||
    value === null ||
    value === ''
  ) {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  const parsed =
    Number(value);

  if (
    !Number.isFinite(
      parsed
    )
  ) {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  return Math.min(
    Math.max(parsed, 0),
    2
  );
}

/**
 * ============================================================================
 * 질문 임베딩
 * ============================================================================
 */

async function createQueryEmbedding(
  userQuery,
  geminiKey
) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/` +
    `models/${EMBEDDING_MODEL}:embedContent` +
    `?key=${encodeURIComponent(
      geminiKey
    )}`;

  const response =
    await fetch(
      url,
      {
        method:
          'POST',

        headers: {
          'Content-Type':
            'application/json',
        },

        body:
          JSON.stringify({
            content: {
              parts: [
                {
                  text:
                    userQuery,
                },
              ],
            },

            outputDimensionality:
              EMBEDDING_DIMENSIONS,
          }),
      }
    );

  const data =
    await response.json();

  if (!response.ok) {
    const apiMessage =
      data?.error?.message ||
      JSON.stringify(data);

    throw new Error(
      `Gemini 임베딩 API 오류: ${apiMessage}`
    );
  }

  const values =
    data?.embedding?.values;

  if (
    !Array.isArray(
      values
    )
  ) {
    throw new Error(
      `Gemini 임베딩 결과가 올바르지 않습니다: ${JSON.stringify(data)}`
    );
  }

  if (
    values.length !==
    EMBEDDING_DIMENSIONS
  ) {
    throw new Error(
      `임베딩 차원이 맞지 않습니다. ` +
      `예상=${EMBEDDING_DIMENSIONS}, ` +
      `실제=${values.length}`
    );
  }

  return values;
}

/**
 * ============================================================================
 * Firestore Vector Search
 * ============================================================================
 */

async function retrieveKnowledge(
  db,
  queryVector,
  topK,
  distanceThreshold
) {
  const vectorQuery =
    db
      .collection(
        'knowledge_chunks'
      )
      .findNearest({
        vectorField:
          'embedding',

        queryVector:
          FieldValue.vector(
            queryVector
          ),

        limit:
          topK,

        distanceMeasure:
          'COSINE',

        distanceResultField:
          'vector_distance',
      });

  const snapshot =
    await vectorQuery.get();

  const searched =
    snapshot.docs
      .map((doc) => {
        const data =
          doc.data();

        return {
          id:
            doc.id,

          text:
            typeof data.text ===
            'string'
              ? data.text
              : '',

          source:
            data.source ||
            null,

          chunk_index:
            typeof data.chunk_index ===
            'number'
              ? data.chunk_index
              : null,

          distance:
            typeof data.vector_distance ===
            'number'
              ? data.vector_distance
              : null,
        };
      })
      .filter(
        (item) =>
          item.text
      );

  const matched =
    searched.filter(
      (item) => {
        if (
          item.distance ===
          null
        ) {
          return true;
        }

        return (
          item.distance <=
          distanceThreshold
        );
      }
    );

  return {
    searched,
    matched,
  };
}

/**
 * ============================================================================
 * RAG Context
 * ============================================================================
 */

function buildContext(
  chunks
) {
  if (!chunks.length) {
    return (
      '[백서 근거]\n' +
      '관련 백서 청크를 찾지 못했습니다.'
    );
  }

  return chunks
    .map(
      (
        chunk,
        index
      ) => {
        const metadata =
          [
            `청크 ${index + 1}`,

            chunk.source
              ? `source=${chunk.source}`
              : null,

            chunk.chunk_index !==
              null
              ? `chunk_index=${chunk.chunk_index}`
              : null,

            chunk.distance !==
              null
              ? `cosine_distance=${chunk.distance.toFixed(6)}`
              : null,
          ]
            .filter(Boolean)
            .join(' | ');

        return (
          `[${metadata}]\n` +
          chunk.text
        );
      }
    )
    .join(
      '\n\n--- 백서 청크 구분 ---\n\n'
    );
}

/**
 * ============================================================================
 * 최종 LLM User Prompt
 * ============================================================================
 *
 * 시스템 프롬프트는 여기에 넣지 않습니다.
 *
 * System:
 *   사용자가 설정창에서 입력한 systemPrompt
 *
 * User:
 *   RAG 검색결과
 *   +
 *   사용자 질문
 * ============================================================================
 */

function buildUserPrompt(
  userQuery,
  retrievedChunks
) {
  return `
[백서 근거]

${buildContext(
  retrievedChunks
)}

[사용자 질문]

${userQuery}
`.trim();
}

/**
 * ============================================================================
 * Gemini
 * ============================================================================
 */

async function generateGeminiAnswer(
  prompt,
  systemPrompt,
  geminiKey
) {
  const ai =
    new GoogleGenAI({
      apiKey:
        geminiKey,
    });

  const response =
    await ai.models.generateContent({
      model:
        MODELS.gemini,

      contents:
        prompt,

      config: {
        systemInstruction:
          systemPrompt,
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
 * ============================================================================
 * GPT-5.6 Luna
 * ============================================================================
 *
 * temperature를 사용하지 않습니다.
 * ============================================================================
 */

async function generateGptAnswer(
  prompt,
  systemPrompt,
  openaiKey
) {
  const openai =
    new OpenAI({
      apiKey:
        openaiKey,
    });

  const response =
    await openai.responses.create({
      model:
        MODELS.gpt,

      instructions:
        systemPrompt,

      input:
        prompt,

      reasoning: {
        effort:
          'medium',
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
 * ============================================================================
 * 전체 RAG 처리
 * ============================================================================
 */

async function runRag({
  message,
  systemPrompt,
  model,
  topK,
  distanceThreshold,
}) {
  const geminiKey =
    process.env.GEMINI_API_KEY;

  if (!geminiKey) {
    throw new Error(
      'GEMINI_API_KEY 환경변수가 설정되지 않았습니다.'
    );
  }

  if (
    model === MODELS.gpt &&
    !process.env.OPENAI_API_KEY
  ) {
    throw new Error(
      'GPT 사용 시 OPENAI_API_KEY 환경변수가 필요합니다.'
    );
  }

  const db =
    getDb();

  /**
   * 1. 질문 embedding
   */
  const queryVector =
    await createQueryEmbedding(
      message,
      geminiKey
    );

  /**
   * 2. Firebase vector search
   */
  const retrieval =
    await retrieveKnowledge(
      db,
      queryVector,
      topK,
      distanceThreshold
    );

  /**
   * 3. RAG User Prompt
   */
  const prompt =
    buildUserPrompt(
      message,
      retrieval.matched
    );

  /**
   * 4. 최종 LLM
   */
  let answer;

  if (
    model === MODELS.gpt
  ) {
    answer =
      await generateGptAnswer(
        prompt,
        systemPrompt,
        process.env.OPENAI_API_KEY
      );
  } else {
    answer =
      await generateGeminiAnswer(
        prompt,
        systemPrompt,
        geminiKey
      );
  }

  return {
    answer,

    model,

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
  };
}

/**
 * ============================================================================
 * Express
 * ============================================================================
 */

app.use(
  express.json({
    limit:
      '1mb',
  })
);

app.use(
  express.urlencoded({
    extended:
      true,
  })
);

/**
 * ============================================================================
 * POST /api/chat
 * ============================================================================
 */

app.post(
  '/api/chat',
  async (
    req,
    res
  ) => {
    try {
      const body =
        parseBody(req);

      const message =
        normalizeUserQuery(
          body.message
        );

      const systemPrompt =
        normalizeSystemPrompt(
          body.systemPrompt
        );

      const model =
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

      const result =
        await runRag({
          message,

          systemPrompt,

          model,

          topK,

          distanceThreshold,
        });

      return res
        .status(200)
        .json(
          result
        );

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
  }
);

/**
 * ============================================================================
 * Health Check
 * ============================================================================
 */

app.get(
  '/api/health',
  (
    req,
    res
  ) => {
    res
      .status(200)
      .json({
        ok:
          true,

        embeddingModel:
          EMBEDDING_MODEL,

        embeddingDimensions:
          EMBEDDING_DIMENSIONS,

        models:
          MODELS,
      });
  }
);

/**
 * ============================================================================
 * 정적 파일
 * ============================================================================
 */

app.use(
  express.static(
    path.join(
      __dirname
    )
  )
);

/**
 * ============================================================================
 * 404
 * ============================================================================
 */

app.use(
  (
    req,
    res
  ) => {
    res
      .status(404)
      .json({
        error:
          'Not found',
      });
  }
);

/**
 * ============================================================================
 * 서버 시작
 * ============================================================================
 */

app.listen(
  PORT,
  () => {
    console.log('');
    console.log(
      '=============================================='
    );
    console.log(
      ' Bloom RAG Chat'
    );
    console.log(
      '=============================================='
    );
    console.log(
      ` Local server : http://localhost:${PORT}`
    );
    console.log(
      ` Health       : http://localhost:${PORT}/api/health`
    );
    console.log('');
    console.log(
      ` Embedding    : ${EMBEDDING_MODEL}`
    );
    console.log(
      ` Dimensions   : ${EMBEDDING_DIMENSIONS}`
    );
    console.log(
      ` Gemini       : ${MODELS.gemini}`
    );
    console.log(
      ` GPT          : ${MODELS.gpt}`
    );
    console.log('');
    console.log(
      ' System Prompt: UI 설정값 사용'
    );
    console.log(
      '=============================================='
    );
    console.log('');
  }
);
