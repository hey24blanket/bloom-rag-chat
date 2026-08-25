const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');
const OpenAI = require('openai');

/**
 * --------------------------------------------------------------------------
 * 고정 설정
 * --------------------------------------------------------------------------
 */

// Firebase에 이미 저장된 백서 embedding과 동일한 공간을 사용해야 합니다.
const EMBEDDING_MODEL = 'gemini-embedding-001';
const EMBEDDING_DIMENSIONS = 768;

// 실제 사용할 답변 모델
const MODELS = {
  gemini: 'gemini-3.1-flash-lite',
  gpt: 'gpt-5.6-luna',
};

// Firestore 검색 설정
const DEFAULT_TOP_K = 5;

// COSINE distance는 작을수록 더 유사합니다.
// 지나치게 먼 문서를 context에 넣지 않기 위한 초기값입니다.
// 실제 서비스 데이터로 테스트하면서 조정합니다.
const DEFAULT_DISTANCE_THRESHOLD = 0.70;

// 답변 생성 기본 설정
const DEFAULT_TEMPERATURE = 0.2;

// 서버가 통제하는 시스템 지침
const SYSTEM_PROMPT = `
너는 "사주그랩 기술 백서"를 근거로 답변하는 RAG(검색증강생성) 어시스턴트다.

반드시 다음 규칙을 지켜라.

1. 제공된 [백서 근거]를 가장 중요한 사실 근거로 사용한다.
2. 백서 근거에 없는 내용을 사실처럼 만들어내지 않는다.
3. 근거가 충분하지 않으면 "제공된 백서에서 확인할 수 없습니다."라고 명확히 말한다.
4. 숫자, 공식, 가중치, 구조, 알고리즘 설명은 백서의 표현과 의미를 최대한 유지한다.
5. 일반적인 상식이나 외부 지식을 보충해서 답할 경우, 백서에 직접 적힌 내용과 구분한다.
6. 질문에 직접 답하고, 불필요한 장황한 설명은 피한다.
7. 검색된 백서 근거를 바탕으로 답변하되, 검색 결과 자체를 사용자에게 그대로 장황하게 복사하지 않는다.
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
        "Vercel 환경변수 'FIREBASE_SERVICE_ACCOUNT'가 설정되지 않았습니다."
      );
    }

    let serviceAccount;

    try {
      serviceAccount =
        typeof rawEnv === 'string' ? JSON.parse(rawEnv) : rawEnv;
    } catch (error) {
      throw new Error(
        `FIREBASE_SERVICE_ACCOUNT JSON 파싱 실패: ${error.message}`
      );
    }

    if (!serviceAccount.project_id) {
      throw new Error('Firebase service account에 project_id가 없습니다.');
    }

    if (!serviceAccount.client_email) {
      throw new Error('Firebase service account에 client_email이 없습니다.');
    }

    if (!serviceAccount.private_key) {
      throw new Error('Firebase service account에 private_key가 없습니다.');
    }

    serviceAccount.private_key = serviceAccount.private_key.replace(
      /\\n/g,
      '\n'
    );

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
    throw new Error('요청 body가 유효한 JSON이 아닙니다.');
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

  // 지나치게 큰 요청 방지
  if (query.length > 10000) {
    throw new Error('질문이 너무 깁니다. 10,000자 이하로 입력해주세요.');
  }

  return query;
}

function normalizeModel(value) {
  if (!value) {
    return MODELS.gemini;
  }

  if (value === MODELS.gemini || value === 'gemini') {
    return MODELS.gemini;
  }

  if (value === MODELS.gpt || value === 'gpt') {
    return MODELS.gpt;
  }

  throw new Error(
    `지원하지 않는 모델입니다. 사용 가능 모델: ${MODELS.gemini}, ${MODELS.gpt}`
  );
}

function normalizeTopK(value) {
  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return DEFAULT_TOP_K;
  }

  return Math.min(Math.max(Math.floor(parsed), 1), 10);
}

function normalizeTemperature(value) {
  if (value === undefined || value === null || value === '') {
    return DEFAULT_TEMPERATURE;
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return DEFAULT_TEMPERATURE;
  }

  return Math.min(Math.max(parsed, 0), 1);
}

function normalizeDistanceThreshold(value) {
  if (value === undefined || value === null || value === '') {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  const parsed = Number(value);

  if (!Number.isFinite(parsed)) {
    return DEFAULT_DISTANCE_THRESHOLD;
  }

  // COSINE distance의 일반적인 범위는 0~2입니다.
  return Math.min(Math.max(parsed, 0), 2);
}

/**
 * --------------------------------------------------------------------------
 * Gemini 임베딩
 * --------------------------------------------------------------------------
 */

async function createQueryEmbedding(userQuery, geminiKey) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/` +
    `models/${EMBEDDING_MODEL}:embedContent?key=${encodeURIComponent(geminiKey)}`;

  const response = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      content: {
        parts: [{ text: userQuery }],
      },
      outputDimensionality: EMBEDDING_DIMENSIONS,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    const apiMessage =
      data?.error?.message || JSON.stringify(data);

    throw new Error(`Gemini 임베딩 API 오류: ${apiMessage}`);
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
      `예상=${EMBEDDING_DIMENSIONS}, 실제=${values.length}`
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
  const vectorQuery = db.collection('knowledge_chunks').findNearest({
    vectorField: 'embedding',
    queryVector: FieldValue.vector(queryVector),
    limit: topK,
    distanceMeasure: 'COSINE',
    distanceResultField: 'vector_distance',
  });

  const snapshot = await vectorQuery.get();

  const results = snapshot.docs
    .map((doc) => {
      const data = doc.data();

      return {
        id: doc.id,
        text: typeof data.text === 'string' ? data.text : '',
        source: data.source || null,
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

  // COSINE: distance가 작을수록 유사
  const filtered = results.filter((item) => {
    if (item.distance === null) {
      // 예상치 못하게 distance가 반환되지 않으면
      // 일단 검색 결과를 살려둡니다.
      return true;
    }

    return item.distance <= distanceThreshold;
  });

  return {
    searched: results,
    matched: filtered,
  };
}

/**
 * --------------------------------------------------------------------------
 * RAG context 구성
 * --------------------------------------------------------------------------
 */

function buildContext(chunks) {
  if (!chunks.length) {
    return '[백서 근거]\n관련 백서 청크를 찾지 못했습니다.';
  }

  const sections = chunks.map((chunk, index) => {
    const metadata = [
      `청크 ${index + 1}`,
      chunk.source ? `source=${chunk.source}` : null,
      chunk.chunk_index !== null
        ? `chunk_index=${chunk.chunk_index}`
        : null,
      chunk.distance !== null
        ? `cosine_distance=${chunk.distance.toFixed(6)}`
        : null,
    ]
      .filter(Boolean)
      .join(' | ');

    return `[${metadata}]\n${chunk.text}`;
  });

  return `[백서 근거]\n\n${sections.join('\n\n--- 백서 청크 구분 ---\n\n')}`;
}

function buildUserPrompt(userQuery, retrievedChunks) {
  const context = buildContext(retrievedChunks);

  return `
${context}

[사용자 질문]
${userQuery}

[답변 지침]
위 백서 근거를 바탕으로 질문에 답변하라.
근거가 충분하지 않으면 "제공된 백서에서 확인할 수 없습니다."라고 답하라.
`.trim();
}

/**
 * --------------------------------------------------------------------------
 * Gemini 답변
 * --------------------------------------------------------------------------
 */

async function generateGeminiAnswer(
  prompt,
  geminiKey,
  temperature
) {
  const ai = new GoogleGenAI({
    apiKey: geminiKey,
  });

  const response = await ai.models.generateContent({
    model: MODELS.gemini,
    contents: prompt,
    config: {
      systemInstruction: SYSTEM_PROMPT,
      temperature,
    },
  });

  const answer = response?.text?.trim();

  if (!answer) {
    throw new Error('Gemini가 빈 답변을 반환했습니다.');
  }

  return answer;
}

/**
 * --------------------------------------------------------------------------
 * GPT-5.6 Luna 답변
 * --------------------------------------------------------------------------
 */

async function generateGptAnswer(
  prompt,
  openaiKey,
  temperature
) {
  const openai = new OpenAI({
    apiKey: openaiKey,
  });

  const response = await openai.responses.create({
    model: MODELS.gpt,
    instructions: SYSTEM_PROMPT,
    input: prompt,
    temperature,
  });

  const answer = response?.output_text?.trim();

  if (!answer) {
    throw new Error('GPT-5.6 Luna가 빈 답변을 반환했습니다.');
  }

  return answer;
}

/**
 * --------------------------------------------------------------------------
 * 메인 API
 * --------------------------------------------------------------------------
 */

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({
      error: 'Method not allowed',
    });
  }

  try {
    const body = parseBody(req);

    const userQuery = normalizeUserQuery(body.message);

    // 브라우저가 보내는 config 중 모델/검색 개수/temperature만 허용합니다.
    // systemPrompt는 절대 클라이언트 값을 신뢰하지 않습니다.
    const selectedModel = normalizeModel(body.model);

    const topK = normalizeTopK(body.topK);
    const temperature = normalizeTemperature(body.temperature);
    const distanceThreshold = normalizeDistanceThreshold(
      body.distanceThreshold
    );

    const geminiKey = process.env.GEMINI_API_KEY;

    if (!geminiKey) {
      return res.status(500).json({
        error:
          "서버 환경변수 'GEMINI_API_KEY'가 설정되지 않았습니다.",
      });
    }

    if (selectedModel === MODELS.gpt && !process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error:
          "서버 환경변수 'OPENAI_API_KEY'가 설정되지 않았습니다.",
      });
    }

    const db = getDb();

    /**
     * 1. 질문 embedding
     *
     * 백서 embedding과 반드시 동일:
     * gemini-embedding-001 / 768차원
     */
    const queryVector = await createQueryEmbedding(
      userQuery,
      geminiKey
    );

    /**
     * 2. Firestore vector search
     */
    const retrieval = await retrieveKnowledge(
      db,
      queryVector,
      topK,
      distanceThreshold
    );

    /**
     * 3. RAG prompt
     */
    const prompt = buildUserPrompt(
      userQuery,
      retrieval.matched
    );

    /**
     * 4. 선택한 LLM으로 답변 생성
     */
    let answer;

    if (selectedModel === MODELS.gpt) {
      answer = await generateGptAnswer(
        prompt,
        process.env.OPENAI_API_KEY,
        temperature
      );
    } else {
      answer = await generateGeminiAnswer(
        prompt,
        geminiKey,
        temperature
      );
    }

    /**
     * 5. 클라이언트에 답변 + 검색 근거 일부 반환
     */
    return res.status(200).json({
      answer,
      model: selectedModel,
      retrieval: {
        topK,
        distanceThreshold,
        searchedCount: retrieval.searched.length,
        matchedCount: retrieval.matched.length,
        chunks: retrieval.matched.map((chunk) => ({
          id: chunk.id,
          source: chunk.source,
          chunk_index: chunk.chunk_index,
          distance: chunk.distance,
          text: chunk.text,
        })),
      },
    });
  } catch (error) {
    console.error('RAG API Error:', error);

    const message =
      error?.message || '서버 내부 오류가 발생했습니다.';

    return res.status(500).json({
      error: message,
    });
  }
};
