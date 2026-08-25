const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');

if (!getApps().length) {
  const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
  initializeApp({ credential: cert(serviceAccount) });
}
const db = getFirestore();

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { message: userQuery, config } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({ error: "Vercel 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다." });
    }

    if (!config) {
      return res.status(400).json({ error: "설정 정보가 전달되지 않았습니다." });
    }

    // GPT 모델 선택 시 분기 처리 영역
    if (config.model === 'gpt-5.6-lunar') {
      // const openaiApiKey = process.env.OPENAI_API_KEY;
      return res.json({ answer: "⚠️ 'GPT 5.6 Lunar' 모델은 현재 Vercel 연동 준비 중입니다. Gemini 모델을 선택해 주세요." });
    }

    const ai = new GoogleGenAI({ apiKey });

    // 1. 임베딩 모델 호출을 위한 목록 조회
    const modelRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?key=${apiKey}`);
    const modelData = await modelRes.json();
    
    if (!modelData.models) {
      throw new Error("Gemini API Key가 올바르지 않거나 모델 목록을 가져올 수 없습니다.");
    }

    const embedModels = modelData.models.filter(m => 
      m.supportedGenerationMethods && m.supportedGenerationMethods.includes('embedContent')
    );
    const embeddingModel = (embedModels.find(m => m.name.includes('text-embedding-004')) || embedModels[0]).name;

    // 2. 질문 벡터화
    const embedRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/${embeddingModel}:embedContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text: userQuery }] }, outputDimensionality: 768 })
    });
    const embedData = await embedRes.json();
    const queryVector = embedData.embedding.values;

    // 3. Firestore Vector DB 검색 (프론트에서 전달된 limit 적용)
    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config.limit || 2,
      distanceMeasure: 'COSINE'
    });

    const snapshot = await vectorQuery.get();
    const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

    // 4. 프롬프트 조합 (프론트에서 전달된 systemPrompt 적용)
    const prompt = `
[시스템 명령어]
${config.systemPrompt}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
    `;

    // 5. LLM 답변 생성 (프론트에서 전달된 model 및 temperature 적용)
    const response = await ai.models.generateContent({
      model: config.model,
      contents: prompt,
      config: { temperature: config.temperature }
    });

    res.json({ answer: response.text });
  } catch (err) {
    console.error("API Error:", err);
    res.status(500).json({ error: err.message || "서버 내부 오류가 발생했습니다." });
  }
};
