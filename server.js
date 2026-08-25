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

app.get('/api/config', (req, res) => {
  res.json(loadConfig());
});

app.post('/api/config', (req, res) => {
  const newConfig = saveConfig(req.body);
  res.json({ success: true, config: newConfig });
});

app.post('/api/chat', async (req, res) => {
  try {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      return res.status(500).json({ error: "GEMINI_API_KEY 환경변수가 설정되지 않았습니다." });
    }

    const userQuery = req.body.message;
    const config = loadConfig();

    // GPT 모델 선택 시 별도 안내
    if (config.model === 'gpt-5.6-lunar') {
      return res.json({ answer: "⚠️ 'GPT 5.6 Lunar' 모델은 Vercel API 엔드포인트 연동 준비 중입니다. Gemini 모델을 선택해 주세요." });
    }

    const ai = new GoogleGenAI({ apiKey });

    // 임베딩 모델 호출
    const url = `https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`;
    const modelRes = await fetch(url);
    const modelData = await modelRes.json();
    
    if (!modelData.models) {
      throw new Error("Gemini API Key가 올바르지 않거나 모델 목록을 가져올 수 없습니다.");
    }

    const embedModels = modelData.models.filter(m => 
      m.supportedGenerationMethods && m.supportedGenerationMethods.includes('embedContent')
    );
    const embeddingModel = (embedModels.find(m => m.name.includes('text-embedding-004')) || embedModels[0]).name;

    // 질문 벡터화
    const embedUrl = `https://generativelanguage.googleapis.com/v1beta/${embeddingModel}:embedContent?key=${apiKey}`;
    const embedRes = await fetch(embedUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text: userQuery }] }, outputDimensionality: 768 })
    });
    const embedData = await embedRes.json();
    const queryVector = embedData.embedding.values;

    // Firestore 검색
    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config.limit || 2,
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

    const response = await ai.models.generateContent({
      model: config.model,
      contents: prompt,
      config: { temperature: config.temperature }
    });

    res.json({ answer: response.text });
  } catch (err) {
    console.error("API 에러 발생:", err);
    res.status(500).json({ error: err.message || "서버 내부 오류가 발생했습니다." });
  }
});

const PORT = 3000;
app.listen(PORT, () => {
  console.log(`🚀 서버 실행 중: http://localhost:${PORT}`);
});
