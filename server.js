const express = require('express');
const path = require('path');
const { initializeApp, cert } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');
const { loadConfig, saveConfig } = require('./config');

const app = express();
app.use(express.json());
app.use(express.static(__dirname));

const serviceAccount = require('./firebase-service-account.json');
initializeApp({ credential: cert(serviceAccount) });
const db = getFirestore();
const apiKey = process.env.GEMINI_API_KEY;
const ai = new GoogleGenAI({ apiKey });

async function getWorkingEmbeddingModel() {
  const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`;
  const res = await fetch(url);
  const data = await res.json();
  const embedModels = data.models.filter(m => 
    m.supportedGenerationMethods && m.supportedGenerationMethods.includes('embedContent')
  );
  return (embedModels.find(m => m.name.includes('text-embedding-004')) || embedModels[0]).name;
}

async function fetchEmbedding(text, modelPath) {
  const url = `https://generativelanguage.googleapis.com/v1beta/${modelPath}:embedContent?key=${apiKey}`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content: { parts: [{ text }] }, outputDimensionality: 768 })
  });
  const data = await response.json();
  return data.embedding.values;
}

app.get('/api/config', (req, res) => {
  res.json(loadConfig());
});

app.post('/api/config', (req, res) => {
  const newConfig = saveConfig(req.body);
  res.json({ success: true, config: newConfig });
});

app.post('/api/chat', async (req, res) => {
  try {
    const userQuery = req.body.message;
    const config = loadConfig();

    const embeddingModel = await getWorkingEmbeddingModel();
    const queryVector = await fetchEmbedding(userQuery, embeddingModel);

    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config.limit,
      distanceMeasure: 'COSINE'
    });

    const snapshot = await vectorQuery.get();
    const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

    const prompt = `
[시스템 명령어]
${config.systemPrompt}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
    `;

    // Gemini 계열 모델 선택 시
    let selectedModel = config.model;
    if (selectedModel === 'gpt-5.6-lunar') {
      // 만약 OpenAI/Vercel 모델 연동 전이라면 알림 처리
      return res.json({ answer: "⚠️ 'GPT 5.6 Lunar' 모델은 현재 Vercel 엔드포인트 연동 준비 중입니다. Gemini 모델을 선택해 주세요." });
    }

    const response = await ai.models.generateContent({
      model: selectedModel,
      contents: prompt,
      config: { temperature: config.temperature }
    });

    res.json({ answer: response.text });
  } catch (err) {
    console.error("Chat API Error:", err);
    // JSON 형태로 안전하게 에러 반환
    res.status(500).json({ error: err.message || "서버 내부 처리 중 오류가 발생했습니다." });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 웹 서버 실행 완료: http://localhost:${PORT}`);
});
