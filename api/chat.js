const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');

// Firebase 중복 초기화 방지
if (!getApps().length) {
  const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT); // Vercel 환경변수 사용
  initializeApp({ credential: cert(serviceAccount) });
}
const db = getFirestore();

module.exports = async (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });

  try {
    const { message: userQuery, config } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    // 1. GPT 선택 시 분기 처리 (OPENAI_API_KEY 사용)
    if (config.model === 'gpt-5.6-lunar') {
      // const openaiKey = process.env.OPENAI_API_KEY;
      return res.json({ answer: "⚠️ GPT 5.6 Lunar 모델 호출 로직 연동 지점입니다. (OPENAI_API_KEY 사용)" });
    }

    // 2. Gemini 임베딩 및 Firestore Vector 검색
    const ai = new GoogleGenAI({ apiKey });
    
    // 질문 벡터화
    const embedRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text: userQuery }] }, outputDimensionality: 768 })
    });
    const embedData = await embedRes.json();
    const queryVector = embedData.embedding.values;

    // Vector DB 검색 (프론트에서 넘겨준 limit 적용)
    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config.limit || 2,
      distanceMeasure: 'COSINE'
    });

    const snapshot = await vectorQuery.get();
    const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

    // 프론트에서 넘겨준 systemPrompt 적용
    const prompt = `
[시스템 명령어]
${config.systemPrompt}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
    `;

    // 프론트에서 넘겨준 model 및 temperature 적용
    const response = await ai.models.generateContent({
      model: config.model,
      contents: prompt,
      config: { temperature: config.temperature }
    });

    res.json({ answer: response.text });
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
};
