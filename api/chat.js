export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method Not Allowed"
    });
  }

  try {
    const { model, messages } = req.body || {};

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: "messages가 필요합니다."
      });
    }

    // 현재는 Luna만 연결
    if (model !== "GPT-5.6 Luna") {
      return res.status(400).json({
        error: "현재는 GPT-5.6 Luna만 연결되어 있습니다."
      });
    }

    if (!process.env.OPENAI_API_KEY) {
      return res.status(500).json({
        error: "OPENAI_API_KEY가 Vercel에 설정되어 있지 않습니다."
      });
    }

    const response = await fetch(
      "https://api.openai.com/v1/responses",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: "gpt-5.6-luna",
          input: messages.map((message) => ({
            role: message.role,
            content: message.content
          }))
        })
      }
    );

    const data = await response.json();

    console.log("OpenAI response:", JSON.stringify(data, null, 2));

    if (!response.ok) {
      return res.status(response.status).json({
        error:
          data?.error?.message ||
          "OpenAI API 요청에 실패했습니다."
      });
    }

    /*
     * Responses API의 실제 응답에서
     * 텍스트를 안전하게 추출합니다.
     */

    let answer = "";

    if (typeof data.output_text === "string") {
      answer = data.output_text;
    }

    if (!answer && Array.isArray(data.output)) {
      for (const item of data.output) {
        if (
          item.type === "message" &&
          Array.isArray(item.content)
        ) {
          for (const content of item.content) {
            if (
              content.type === "output_text" &&
              typeof content.text === "string"
            ) {
              answer += content.text;
            }
          }
        }
      }
    }

    if (!answer) {
      console.error(
        "No text found in OpenAI response:",
        JSON.stringify(data, null, 2)
      );

      return res.status(500).json({
        error: "OpenAI 응답은 받았지만 텍스트를 찾을 수 없습니다."
      });
    }

    return res.status(200).json({
      answer,
      model: "GPT-5.6 Luna"
    });

  } catch (error) {
    console.error("Server Error:", error);

    return res.status(500).json({
      error:
        error?.message ||
        "서버에서 오류가 발생했습니다."
    });
  }
}
