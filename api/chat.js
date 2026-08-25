export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method Not Allowed"
    });
  }

  try {
    const {
      model,
      messages
    } = req.body || {};

    if (
      !Array.isArray(messages) ||
      messages.length === 0
    ) {
      return res.status(400).json({
        error: "messages가 필요합니다."
      });
    }


    /* =========================================
       1. GPT-5.6 Luna
    ========================================== */

    if (model === "GPT-5.6 Luna") {

      if (!process.env.OPENAI_API_KEY) {
        return res.status(500).json({
          error:
            "OPENAI_API_KEY가 Vercel에 설정되어 있지 않습니다."
        });
      }


      const response = await fetch(
        "https://api.openai.com/v1/responses",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",

            "Authorization":
              `Bearer ${process.env.OPENAI_API_KEY}`
          },

          body: JSON.stringify({
            model: "gpt-5.6-luna",

            input: messages.map(
              (message) => ({
                role:
                  message.role === "assistant"
                    ? "assistant"
                    : "user",

                content:
                  message.content
              })
            )
          })
        }
      );


      const data =
        await response.json();


      console.log(
        "OpenAI response:",
        JSON.stringify(
          data,
          null,
          2
        )
      );


      if (!response.ok) {

        return res.status(
          response.status
        ).json({
          error:
            data?.error?.message ||
            "OpenAI API 요청에 실패했습니다."
        });

      }


      /*
       * Responses API 텍스트 추출
       */

      let answer = "";


      if (
        typeof data.output_text ===
        "string"
      ) {
        answer =
          data.output_text;
      }


      if (
        !answer &&
        Array.isArray(data.output)
      ) {

        for (
          const item
          of data.output
        ) {

          if (
            item.type === "message" &&
            Array.isArray(
              item.content
            )
          ) {

            for (
              const content
              of item.content
            ) {

              if (
                content.type ===
                  "output_text" &&
                typeof content.text ===
                  "string"
              ) {

                answer +=
                  content.text;

              }

            }
          }
        }
      }


      if (!answer) {

        console.error(
          "OpenAI text not found:",
          JSON.stringify(
            data,
            null,
            2
          )
        );


        return res.status(500).json({
          error:
            "OpenAI 응답은 받았지만 텍스트를 찾을 수 없습니다."
        });

      }


      return res.status(200).json({
        answer,
        model: "GPT-5.6 Luna"
      });
    }


    /* =========================================
       2. Gemini 3.1 Flash-Lite
    ========================================== */

    if (
      model ===
      "Gemini 3.1 Flash-Lite"
    ) {

      if (!process.env.GEMINI_API_KEY) {

        return res.status(500).json({
          error:
            "GEMINI_API_KEY가 Vercel에 설정되어 있지 않습니다."
        });

      }


      /*
       * Gemini API에 보낼 대화 형식으로 변환
       *
       * user      → user
       * assistant → model
       */

      const contents =
        messages.map(
          (message) => ({
            role:
              message.role === "assistant"
                ? "model"
                : "user",

            parts: [
              {
                text:
                  message.content
              }
            ]
          })
        );


      const response =
        await fetch(
          "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-lite:generateContent",
          {
            method: "POST",

            headers: {
              "Content-Type":
                "application/json",

              "x-goog-api-key":
                process.env.GEMINI_API_KEY
            },

            body: JSON.stringify({
              contents
            })
          }
        );


      const data =
        await response.json();


      console.log(
        "Gemini response:",
        JSON.stringify(
          data,
          null,
          2
        )
      );


      if (!response.ok) {

        return res.status(
          response.status
        ).json({
          error:
            data?.error?.message ||
            "Gemini API 요청에 실패했습니다."
        });

      }


      /*
       * Gemini 응답 텍스트 추출
       */

      const answer =
        data?.candidates?.[0]?.content?.parts
          ?.map(
            (part) =>
              part?.text || ""
          )
          .join("")
          .trim();


      if (!answer) {

        console.error(
          "Gemini text not found:",
          JSON.stringify(
            data,
            null,
            2
          )
        );


        return res.status(500).json({
          error:
            "Gemini 응답은 받았지만 텍스트를 찾을 수 없습니다."
        });

      }


      return res.status(200).json({
        answer,

        model:
          "Gemini 3.1 Flash-Lite"
      });
    }


    /* =========================================
       3. 지원하지 않는 모델
    ========================================== */

    return res.status(400).json({
      error:
        `지원하지 않는 모델입니다: ${model}`
    });

  } catch (error) {

    console.error(
      "Server Error:",
      error
    );


    return res.status(500).json({
      error:
        error?.message ||
        "서버에서 오류가 발생했습니다."
    });
  }
}
