const fs = require('fs');
const path = require('path');

// 저장 파일 경로 지정 (동일 폴더 내 config.json)
const CONFIG_FILE_PATH = path.join(__dirname, 'config.json');

// 기본 설정값 (Default Configuration)
const DEFAULT_CONFIG = {
  // 1. LLM & 검색 관련 동적 파라미터 (즉시 반영)
  systemPrompt: "당신은 사주그랩 기술 백서 기반 전문가입니다. 아래 제공된 [백서 내용]만을 참고하여 질문에 답하세요.",
  model: "gemini-3.1-flash-lite",
  limit: 2,
  temperature: 0.2,
  similarityThreshold: 0.0,

  // 2. 백서 재처리가 필요한 정적 파라미터 (참고용)
  chunkSize: 800,
  chunkOverlap: 100
};

/**
 * config.json 파일에서 설정 불러오기
 * (파일이 없으면 DEFAULT_CONFIG로 신규 생성)
 */
function loadConfig() {
  try {
    if (!fs.existsSync(CONFIG_FILE_PATH)) {
      saveConfig(DEFAULT_CONFIG);
      return DEFAULT_CONFIG;
    }
    const data = fs.readFileSync(CONFIG_FILE_PATH, 'utf8');
    return JSON.parse(data);
  } catch (error) {
    console.error('❌ 설정 파일 읽기 실패. 기본 설정을 사용합니다:', error.message);
    return DEFAULT_CONFIG;
  }
}

/**
 * config.json 파일에 설정 저장하기
 */
function saveConfig(newConfig) {
  try {
    const updatedConfig = { ...loadConfig(), ...newConfig };
    fs.writeFileSync(CONFIG_FILE_PATH, JSON.stringify(updatedConfig, null, 2), 'utf8');
    console.log('✅ 설정이 config.json에 정상적으로 저장되었습니다.');
    return updatedConfig;
  } catch (error) {
    console.error('❌ 설정 파일 저장 실패:', error.message);
    throw error;
  }
}

module.exports = {
  loadConfig,
  saveConfig,
  CONFIG_FILE_PATH
};
