const express = require('express');
const path = require('path');
const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');
const OpenAI = require('openai');

const app = express();
const PORT = Number(process.env.PORT) || 3000;

/**
 * --------------------------------------------------------------------------
 * 고정 설정
 * --------------------------------------------------------------------------
 */

const EMBEDDING_MODEL = 'gemini-embedding-001';
const EMBEDDING_DIMENSIONS = 768;

const MODELS = {
  gemini: 'gemini-3.1-flash-lite',
  gpt: 'gpt-5.6-luna',
};

const DEFAULT_TOP_K = 5;
const DEFAULT_DISTANCE_THRESHOLD = 0.70;
const DEFAULT_TEMPERATURE = 0.2;

const SYSTEM_PROMPT = `
너는 "사주그랩 기술 백서"를 근거로 답변하는 RAG 어시스턴트다.

반드시 다음 규칙을 지켜라.

1. 제공된 [백서 근거]를 가장 중요한 사실 근거로 사용한다.
2. 백서 근거에 없는 내용을 사실처럼 만들어내지 않는다.
3. 근거가 충분하지 않으면 "제공된 백서에서 확인할 수 없습니다."라고 명확히 말한다.
4. 숫자, 공식, 가중치, 구조, 알고리즘 설명은 백서의 표현과 의미를 최대한 유지한다.
5. 일반적인 상식이나 외부 지식을 보충해서 답할 경우, 백서에 직접 적힌 내용과 구분한다.
6. 질문에 직접 답하고, 불필요하게 장황하게 설명하지 않는다.
`.trim();

/**
 * --------------------------------------------------------------------------
 * Firebase
 * --------------------------------------------------------------------------
 */

function getDb() {
  if (!getApps().length) {
    const rawEnv = process.env.FIREBASE_SERVICE_ACCOUNT;

    if (!rawEnv) {
      throw new Error(
        "환경변수 'FIREBASE_SERVICE_ACCOUNT'가 설정되지 않았습니다."
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
      throw new Error('Firebase service account의 project_id가 없습니다.');
    }

    if (!serviceAccount.client_email) {
      throw new Error('Firebase service account의 client_email이 없습니다.');
    }

    if (!serviceAccount.private_key) {
      throw new Error('Firebase service account의 private_key가 없습니다.');
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

function normalizeQuery(value) {
  if (typeof value !== 'string') {
    throw new Error('message는 문자열이어야 합니다.');
  }

  const query = value.trim();

  if (!query) {
    throw new Error('질문을 입력해주세요.');
  }

  if (query.length > 10000) {
    throw new Error('질문이 너무 깁니다. 10,000자 이하로 입력해주세요.');
  }

  return query;
}

function normalizeModel(value) {
  if (!value) {
    return MODELS.gemini;
  }

  if (value === 'gemini' || value === MODELS.gemini) {
    return MODELS.gemini;
  }

  if (value === 'gpt' || value === MODELS.gpt) {
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

  return Math.min(Math.max(parsed, 0), 2);
}

/**
 * --------------------------------------------------------------------------
 * Gemini embedding
 * --------------------------------------------------------------------------
 */

async function createQueryEmbedding(query, geminiKey) {
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
        parts: [{ text: query }],
      },
      outputDimensionality: EMBEDDING_DIMENSIONS,
    }),
  });

  const data = await response.json();

  if (!response.ok) {
    const message = data?.error?.message || JSON.stringify(data);
    throw new Error(`Gemini 임베딩 API 오류: ${message}`);
  }

  const values = data?.embedding?.values;

  if (!Array.isArray(values)) {
    throw new Error(
      `Gemini 임베딩 결과가 올바르지 않습니다: ${JSON.stringify(data)}`
    );
  }

  if (values.length !== EMBEDDING_DIMENSIONS) {
    throw new Error(
      `임베딩 차원 불일치: 예상=${EMBEDDING_DIMENSIONS}, 실제=${values.length}`
    );
  }

  return values;
}

/**
 * --------------------------------------------------------------------------
 * Firestore 검색
 * --------------------------------------------------------------------------
 */

async function retrieveKnowledge(
  db,
  queryVector,
  topK,
  distanceThreshold
) {
  const query = db.collection('knowledge_chunks').findNearest({
    vectorField: 'embedding',
    queryVector: FieldValue.vector(queryVector),
    limit: topK,
    distanceMeasure: 'COSINE',
    distanceResultField: 'vector_distance',
  });

  const snapshot = await query.get();

  const searched = snapshot.docs
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

  const matched = searched.filter((item) => {
    if (item.distance === null) {
      return true;
    }

    return item.distance <= distanceThreshold;
  });

  return {
    searched,
    matched,
  };
}

/**
 * --------------------------------------------------------------------------
 * RAG Prompt
 * --------------------------------------------------------------------------
 */

function buildContext(chunks) {
  if (!chunks.length) {
    return '[백서 근거]\n관련 백서 청크를 찾지 못했습니다.';
  }

  return chunks
    .map((chunk, index) => {
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
    })
    .join('\n\n--- 백서 청크 구분 ---\n\n');
}

function buildPrompt(query, chunks) {
  return `
[백서 근거]

${buildContext(chunks)}

[사용자 질문]

${query}

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
 * RAG 처리
 * --------------------------------------------------------------------------
 */

async function runRag({
  message,
  model,
  topK,
  distanceThreshold,
  temperature,
}) {
  const geminiKey = process.env.GEMINI_API_KEY;

  if (!geminiKey) {
    throw new Error(
      "GEMINI_API_KEY 환경변수가 설정되지 않았습니다."
    );
  }

  if (model === MODELS.gpt && !process.env.OPENAI_API_KEY) {
    throw new Error(
      "GPT 사용 시 OPENAI_API_KEY 환경변수가 필요합니다."
    );
  }

  const db = getDb();

  // 1. 질문 embedding
  const queryVector = await createQueryEmbedding(
    message,
    geminiKey
  );

  // 2. Firebase vector search
  const retrieval = await retrieveKnowledge(
    db,
    queryVector,
    topK,
    distanceThreshold
  );

  // 3. RAG prompt
  const prompt = buildPrompt(
    message,
    retrieval.matched
  );

  // 4. LLM
  let answer;

  if (model === MODELS.gpt) {
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

  return {
    answer,
    model,
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
  };
}

/**
 * --------------------------------------------------------------------------
 * Express middleware
 * --------------------------------------------------------------------------
 */

app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true }));

/**
 * --------------------------------------------------------------------------
 * Chat API
 * --------------------------------------------------------------------------
 */

app.post('/api/chat', async (req, res) => {
  try {
    const message = normalizeQuery(req.body?.message);
    const model = normalizeModel(req.body?.model);

    const topK = normalizeTopK(req.body?.topK);

    const distanceThreshold = normalizeDistanceThreshold(
      req.body?.distanceThreshold
    );

    const temperature = normalizeTemperature(
      req.body?.temperature
    );

    const result = await runRag({
      message,
      model,
      topK,
      distanceThreshold,
      temperature,
    });

    return res.status(200).json(result);
  } catch (error) {
    console.error('RAG API Error:', error);

    return res.status(500).json({
      error:
        error?.message ||
        '서버 내부 오류가 발생했습니다.',
    });
  }
});

/**
 * --------------------------------------------------------------------------
 * 정적 파일
 * --------------------------------------------------------------------------
 */

app.use(express.static(path.join(__dirname)));

/**
 * --------------------------------------------------------------------------
 * Health check
 * --------------------------------------------------------------------------
 */

app.get('/api/health', (req, res) => {
  res.status(200).json({
    ok: true,
    embeddingModel: EMBEDDING_MODEL,
    embeddingDimensions: EMBEDDING_DIMENSIONS,
    models: MODELS,
  });
});

/**
 * --------------------------------------------------------------------------
 * 404
 * --------------------------------------------------------------------------
 */

app.use((req, res) => {
  res.status(404).json({
    error: 'Not found',
  });
});

/**
 * --------------------------------------------------------------------------
 * Start
 * --------------------------------------------------------------------------
 */

app.listen(PORT, () => {
  console.log('');
  console.log('==============================================');
  console.log(' Bloom RAG Chat');
  console.log('==============================================');
  console.log(` Local server : http://localhost:${PORT}`);
  console.log(` Health       : http://localhost:${PORT}/api/health`);
  console.log('');
  console.log(` Embedding    : ${EMBEDDING_MODEL}`);
  console.log(` Dimensions   : ${EMBEDDING_DIMENSIONS}`);
  console.log(` Gemini       : ${MODELS.gemini}`);
  console.log(` GPT          : ${MODELS.gpt}`);
  console.log('==============================================');
  console.log('');
});
