const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');

// Firebase 안전 초기화 함수
function getDb() {
  if (!getApps().length) {
    const rawEnv = process.env.FIREBASE_SERVICE_ACCOUNT;
    if (!rawEnv) {
      throw new Error("Vercel 환경변수 'FIREBASE_SERVICE_ACCOUNT'가 설정되지 않았습니다.");
    }

    let serviceAccount;
    try {
      serviceAccount = typeof rawEnv === 'string' ? JSON.parse(rawEnv) : rawEnv;
    } catch (e) {
      throw new Error("FIREBASE_SERVICE_ACCOUNT JSON 파싱 실패: " + e.message);
    }

    // Vercel 환경변수의 private_key 줄바꿈(\\n -> \n) 자동 보정
    if (serviceAccount.private_key) {
      serviceAccount.private_key = serviceAccount.private_key.replace(/\\n/g, '\n');
    }

    initializeApp({ credential: cert(serviceAccount) });
  }
  return getFirestore();
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const db = getDb();
    const { message: userQuery, config } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
      return res.status(500).json({ error: "Vercel 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다." });
    }

    // GPT 선택 시 안심 안내
    if (config?.model === 'gpt-5.6-lunar') {
      return res.json({ answer: "⚠️ 'GPT 5.6 Lunar' 모델은 현재 연동 준비 중입니다. 설정(⚙️) 모달에서 'Gemini 3.1 Flash-Lite'를 선택해 주세요." });
    }

    const ai = new GoogleGenAI({ apiKey });

    // 1. 임베딩 생성
    const embedRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${apiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text: userQuery }] }, outputDimensionality: 768 })
    });
    const embedData = await embedRes.json();
    
    if (!embedData.embedding?.values) {
      throw new Error("임베딩 생성 실패: " + JSON.stringify(embedData));
    }
    const queryVector = embedData.embedding.values;

    // 2. Vector DB 유사도 검색
    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config?.limit || 2,
      distanceMeasure: 'COSINE'
    });

    const snapshot = await vectorQuery.get();
    const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

    // 3. 프롬프트 생성 및 답변 요청
    const prompt = `
[시스템 명령어]
${config?.systemPrompt || '백서 내용을 바탕으로 답변하세요.'}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
    `;

    const response = await ai.models.generateContent({
      model: config?.model || 'gemini-3.1-flash-lite',
      contents: prompt,
      config: { temperature: config?.temperature ?? 0.2 }
    });

    res.json({ answer: response.text });
  } catch (err) {
    console.error("Vercel Function Error:", err);
    res.status(500).json({ error: err.message || "서버 내부 처리 오류가 발생했습니다." });
  }
};
