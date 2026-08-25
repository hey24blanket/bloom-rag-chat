const { initializeApp, cert, getApps } = require('firebase-admin/app');
const { getFirestore, FieldValue } = require('firebase-admin/firestore');
const { GoogleGenAI } = require('@google/genai');

// Firebase 안전 초기화
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
    const geminiKey = process.env.GEMINI_API_KEY;
    const openaiKey = process.env.OPENAI_API_KEY;

    if (!geminiKey) {
      return res.status(500).json({ error: "Vercel 환경변수에 GEMINI_API_KEY가 설정되지 않았습니다." });
    }

    // 1. Vector DB 검색용 임베딩 생성 (Gemini 사용)
    const embedRes = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key=${geminiKey}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: { parts: [{ text: userQuery }] }, outputDimensionality: 768 })
    });
    const embedData = await embedRes.json();

    if (!embedData.embedding?.values) {
      throw new Error("임베딩 생성 실패: " + (embedData.error?.message || JSON.stringify(embedData)));
    }
    const queryVector = embedData.embedding.values;

    // 2. Firestore Vector DB 유사도 검색
    const vectorQuery = db.collection('knowledge_chunks').findNearest({
      vectorField: 'embedding',
      queryVector: FieldValue.vector(queryVector),
      limit: config?.limit || 2,
      distanceMeasure: 'COSINE'
    });

    const snapshot = await vectorQuery.get();
    const retrievedContexts = snapshot.docs.map(doc => doc.data().text);

    // 3. RAG 프롬프트 구성
    const prompt = `
[시스템 명령어]
${config?.systemPrompt || '백서 내용을 바탕으로 답변하세요.'}

[백서 내용]
${retrievedContexts.join('\n\n--- 청크 구분선 ---\n\n')}

[사용자 질문]
${userQuery}
    `;

    // 4. 선택된 모델별 LLM 답변 생성
    const selectedModel = config?.model || 'gemini-3.1-flash-lite';

    if (selectedModel.startsWith('gpt') || selectedModel.includes('lunar')) {
      if (!openaiKey) {
        return res.status(500).json({ error: "Vercel 환경변수에 OPENAI_API_KEY가 설정되지 않았습니다." });
      }

      // OpenAI API 호출 (선택된 'gpt-5.6-lunar' 모델명을 그대로 전달)
      const gptRes = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${openaiKey}`
        },
        body: JSON.stringify({
          model: selectedModel, // 'gpt-5.6-lunar' 전달
          messages: [{ role: 'user', content: prompt }],
          temperature: config?.temperature ?? 0.2
        })
      });

      const gptData = await gptRes.json();
      if (gptData.error) {
        throw new Error(`OpenAI API 오류: ${gptData.error.message}`);
      }

      return res.json({ answer: gptData.choices[0]?.message?.content || "답변을 가져올 수 없습니다." });

    } else {
      // Gemini API 호출
      const ai = new GoogleGenAI({ apiKey: geminiKey });
      const response = await ai.models.generateContent({
        model: selectedModel,
        contents: prompt,
        config: { temperature: config?.temperature ?? 0.2 }
      });

      return res.json({ answer: response.text });
    }

  } catch (err) {
    console.error("Vercel Function Error:", err);
    return res.status(500).json({ error: err.message || "서버 내부 오류가 발생했습니다." });
  }
};
