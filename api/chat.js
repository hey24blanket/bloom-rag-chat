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

/**
 * ============================================================================
 * 고정 설정
 * ============================================================================
 *
 * 현재 Firebase knowledge_chunks에 저장된 embedding 기준:
 *
 *   모델: gemini-embedding-001
 *   차원: 768
 *
 * 질문 embedding도 반드시 동일한 모델/차원을 사용합니다.
 */

const EMBEDDING_MODEL =
  'gemini-embedding-001';

const EMBEDDING_DIMENSIONS =
  768;

/**
 * 실제 답변 생성에 사용할 모델
 */
const MODELS = {
  gemini:
    'gemini-3.1-flash-lite',

  gpt:
    'gpt-5.6-luna',
};

/**
 * 기본 RAG 설정
 */
const DEFAULT_TOP_K =
  5;

const DEFAULT_DISTANCE_THRESHOLD =
  0.70;


/**
 * ============================================================================
 * 시간 측정
 * ============================================================================
 *
 * performance.now()를 사용할 수 있는 Node 환경을 기준으로 합니다.
 *
 * 반환 단위는 milliseconds(ms)입니다.
 */

function now() {
  return performance.now();
}


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

    /**
     * 환경변수의 \n 문자를 실제 줄바꿈으로 변환
     */
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
 * 요청 Body
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
    query.length >
    10000
  ) {
    throw new Error(
      '질문이 너무 깁니다. 10,000자 이하로 입력해주세요.'
    );
  }

  return query;
}


/**
 * --------------------------------------------------------------------------
 * System Prompt
 * --------------------------------------------------------------------------
 *
 * 시스템 프롬프트는 서버에 하드코딩하지 않습니다.
 * index.html의 설정창에서 저장한 값을 전달받습니다.
 */

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
    prompt.length >
    30000
  ) {
    throw new Error(
      '시스템 프롬프트가 너무 깁니다. 30,000자 이하로 입력해주세요.'
    );
  }

  return prompt;
}


/**
 * --------------------------------------------------------------------------
 * Model
 * --------------------------------------------------------------------------
 */

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


/**
 * --------------------------------------------------------------------------
 * Top K
 * --------------------------------------------------------------------------
 */

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


/**
 * --------------------------------------------------------------------------
 * Cosine Distance Threshold
 * --------------------------------------------------------------------------
 */

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

  /**
   * 일반적인 cosine distance 범위:
   * 0 ~ 2
   *
   * 낮을수록 유사
   */
  return Math.min(
    Math.max(parsed, 0),
    2
  );
}


/**
 * ============================================================================
 * Query Embedding
 * ============================================================================
 *
 * Firebase 백서 embedding과 동일한:
 *
 *   gemini-embedding-001
 *   768 dimensions
 *
 * 를 사용합니다.
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

        /**
         * 실제 vector distance를
         * 결과 데이터에 추가
         */
        distanceResultField:
          'vector_distance',
      });

  const snapshot =
    await vectorQuery.get();

  const searched =
    snapshot.docs
      .map(
        (doc) => {
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
        }
      )
      .filter(
        (item) =>
          item.text
      );

  /**
   * Cosine distance가 threshold보다 작은
   * 결과만 최종적으로 사용
   */
  const matched =
    searched.filter(
      (item) => {
        /**
         * 혹시 Firestore에서 distance가
         * 반환되지 않는 예외 상황이라면
         * 검색 결과를 일단 유지
         */
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
            .join(
              ' | '
            );

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
 * Final User Prompt
 * ============================================================================
 *
 * System Prompt는 별도로 LLM의
 * system / instructions에 들어갑니다.
 *
 * 여기에는:
 *
 *   RAG 검색 결과
 *   +
 *   사용자 질문
 *
 * 만 들어갑니다.
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
 * Gemini Answer
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
        /**
         * 사용자가 설정창에서
         * 저장한 시스템 프롬프트
         */
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
 * GPT-5.6 Luna Answer
 * ============================================================================
 *
 * temperature는 사용하지 않습니다.
 *
 * reasoning.effort:
 *   medium
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

      /**
       * 사용자 설정창에서
       * 전달된 System Prompt
       */
      instructions:
        systemPrompt,

      /**
       * RAG 검색 결과 +
       * 사용자 질문
       */
      input:
        prompt,

      /**
       * GPT-5.6 Luna reasoning
       */
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
 * 메인 API
 * ============================================================================
 */

module.exports =
  async (
    req,
    res
  ) => {

    /**
     * ------------------------------------------------------------------------
     * Method Check
     * ------------------------------------------------------------------------
     */

    if (
      req.method !==
      'POST'
    ) {
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

    /**
     * 서버 전체 처리 시작
     */
    const totalStart =
      now();

    try {

      /**
       * ----------------------------------------------------------------------
       * 1. Request Parsing
       * ----------------------------------------------------------------------
       */

      const body =
        parseBody(req);

      const userQuery =
        normalizeUserQuery(
          body.message
        );

      const systemPrompt =
        normalizeSystemPrompt(
          body.systemPrompt
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


      /**
       * ----------------------------------------------------------------------
       * API Keys
       * ----------------------------------------------------------------------
       */

      const geminiKey =
        process.env.GEMINI_API_KEY;

      if (!geminiKey) {
        return res
          .status(500)
          .json({
            error:
              'GEMINI_API_KEY 환경변수가 설정되지 않았습니다.',
          });
      }

      if (
        selectedModel ===
          MODELS.gpt &&
        !process.env.OPENAI_API_KEY
      ) {
        return res
          .status(500)
          .json({
            error:
              'OPENAI_API_KEY 환경변수가 설정되지 않았습니다.',
          });
      }


      /**
       * ----------------------------------------------------------------------
       * Firebase
       * ----------------------------------------------------------------------
       */

      const db =
        getDb();


      /**
       * ======================================================================
       * STEP 1
       * Query Embedding
       * ======================================================================
       */

      const embeddingStart =
        now();

      const queryVector =
        await createQueryEmbedding(
          userQuery,
          geminiKey
        );

      const embeddingMs =
        now() -
        embeddingStart;


      /**
       * ======================================================================
       * STEP 2
       * Firebase Vector Search
       * ======================================================================
       */

      const retrievalStart =
        now();

      const retrieval =
        await retrieveKnowledge(
          db,

          queryVector,

          topK,

          distanceThreshold
        );

      const retrievalMs =
        now() -
        retrievalStart;


      /**
       * ======================================================================
       * STEP 3
       * Build RAG User Prompt
       * ======================================================================
       *
       * 이 시간은 LLM 시간에 포함하지 않습니다.
       *
       * 실제로 중요한 병목은:
       *
       *   Embedding
       *   Firebase RAG
       *   LLM
       *
       * 세 구간입니다.
       */

      const prompt =
        buildUserPrompt(
          userQuery,
          retrieval.matched
        );


      /**
       * ======================================================================
       * STEP 4
       * LLM Generation
       * ======================================================================
       */

      const llmStart =
        now();

      let answer;

      if (
        selectedModel ===
        MODELS.gpt
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

      const llmMs =
        now() -
        llmStart;


      /**
       * ======================================================================
       * TOTAL
       * ======================================================================
       */

      const totalMs =
        now() -
        totalStart;


      /**
       * ======================================================================
       * Console Debug
       * ======================================================================
       *
       * Vercel 로그 / 로컬 로그에서
       * 병목을 바로 확인할 수 있습니다.
       */

      console.log(
        '[RAG Timing]',
        {
          model:
            selectedModel,

          embeddingMs:
            Number(
              embeddingMs.toFixed(
                2
              )
            ),

          retrievalMs:
            Number(
              retrievalMs.toFixed(
                2
              )
            ),

          llmMs:
            Number(
              llmMs.toFixed(
                2
              )
            ),

          totalMs:
            Number(
              totalMs.toFixed(
                2
              )
            ),

          topK,

          searchedCount:
            retrieval.searched.length,

          matchedCount:
            retrieval.matched.length,
        }
      );


      /**
       * ======================================================================
       * Response
       * ======================================================================
       */

      return res
        .status(200)
        .json({

          /**
           * 최종 LLM 답변
           */
          answer,

          /**
           * 사용 모델
           */
          model:
            selectedModel,

          /**
           * RAG 검색 결과
           */
          retrieval: {

            topK,

            distanceThreshold,

            searchedCount:
              retrieval.searched.length,

            matchedCount:
              retrieval.matched.length,

            chunks:
              retrieval.matched.map(
                (
                  chunk
                ) => ({

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

          /**
           * 처리 시간
           *
           * index.html이 이 값을 받아서:
           *
           * 임베딩
           * Firebase RAG
           * LLM
           * 전체
           *
           * 를 화면에 표시합니다.
           */
          timing: {

            embeddingMs:
              Number(
                embeddingMs.toFixed(
                  2
                )
              ),

            retrievalMs:
              Number(
                retrievalMs.toFixed(
                  2
                )
              ),

            llmMs:
              Number(
                llmMs.toFixed(
                  2
                )
              ),

            totalMs:
              Number(
                totalMs.toFixed(
                  2
                )
              ),

          },

        });

    } catch (
      error
    ) {

      /**
       * 전체 처리 시간
       * 오류가 난 경우에도 로그로 남김
       */
      const totalMs =
        now() -
        totalStart;

      console.error(
        '[RAG API Error]',
        error
      );

      console.error(
        '[RAG Error Timing]',
        {
          totalMs:
            Number(
              totalMs.toFixed(
                2
              )
            ),
        }
      );

      return res
        .status(500)
        .json({

          error:
            error?.message ||
            '서버 내부 오류가 발생했습니다.',

          timing: {

            totalMs:
              Number(
                totalMs.toFixed(
                  2
                )
              ),

          },

        });

    }

  };
