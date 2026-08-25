const readline = require('readline');
const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');
const { loadConfig } = require('./config'); // config.js 모듈 로드

// 1. Firebase 및 Gemini API 초기화
const serviceAccount = require('./firebase-service-account.json');
initializeApp({ credential: cert(serviceAccount) });

const db = getFirestore();
const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({ apiKey });

// 2. 사용 가능한 임베딩 모델 자동 감지
async function getWorkingEmbeddingModel() {
  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`;
  const res = await fetch(url);
  const data = await res.json();
  const embedModels = data.models.filter(m => 
    m.supportedGenerationMethods && m.supportedGenerationMethods.includes('embedContent')
  );
  const preferred = embedModels.find(m => m.name.includes('text-embedding-004')) || embedModels[0];
  return preferred.name;
}

// 3. 질문 벡터화 (768차원)
async function fetchEmbedding(text, modelPath) {
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:embedContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      content: { parts: [{ text }] },
      outputDimensionality: 768
    })
  });
  const data = await response.json();
  return data.embedding.values;
}

// 4. RAG 검색 및 답변 생성 (config.json 파일 설정 연동)
async function askRAG(userQuery) {
  // 질문이 들어올 때마다 최신 config.json 파일의 설정을 불러옵니다.
  const config = loadConfig();

  const embeddingModel = await getWorkingEmbeddingModel();
  const queryVector = await fetchEmbedding(userQuery, embeddingModel);

  // config.json에 지정된 limit 개수만큼 Firestore 코사인 검색
  const vectorQuery = db.collection('knowledge_chunks').findNearest({
    vectorField: 'embedding',
    queryVector: FieldValue.vector(queryVector),
    limit: config.limit,
    distanceMeasure: 'COSINE'
  });

  const snapshot = await vectorQuery.get();
  const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

  // config.json의 systemPrompt 사용
  const prompt = `
[시스템 명령어]
${config.systemPrompt}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
  `;

  // config.json의 model 및 temperature 적용
  const response = await ai.models.generateContent({
    model: config.model,
    contents: prompt,
    config: {
      temperature: config.temperature
    }
  });

  return response.text;
}

// 5. 터미널 대화 루프 (CLI Interface)
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log('✨ 사주그랩 백서 RAG 챗봇 연결 완료! (종료하려면 exit 입력)');

function startChat() {
  rl.question('\n❓ 질문: ', async (userInput) => {
    if (userInput.trim().toLowerCase() === 'exit') {
      console.log('👋 챗봇을 종료합니다.');
      rl.close();
      process.exit(0);
    }

    try {
      process.stdout.write('🔍 백서 검색 및 Gemini 답변 생성 중...');
      const answer = await askRAG(userInput);
      console.log('\r                                         ');
      console.log(`🤖 [사주그랩 AI]:\n${answer}`);
    } catch (err) {
      console.error('\n❌ 에러 발생:', err.message);
    }

    startChat();
  });
}

startChat();
