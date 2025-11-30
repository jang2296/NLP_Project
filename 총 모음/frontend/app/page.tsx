'use client';

import React, { useState, useEffect, useRef } from 'react';

// --- API 설정 ---
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://k-euphemism-api-245053314944.asia-northeast3.run.app';

// --- 데이터 구조 정의 ---
interface AnalysisSegment {
  text: string;
  type: 'ORG' | 'PER' | 'LOC' | 'normal';
  confidence: number;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string | AnalysisSegment[];
}

interface HistoryItem {
  id: string;
  originalText: string;
  result?: AnalysisSegment[];
  timestamp: string;
}

// API 응답 타입 정의
interface APIDetection {
  text: string;
  type: string;
  start: number;
  end: number;
  entity?: string;
  confidence: number;
  alternatives?: Array<{ name: string; confidence: number }>;
}

interface APIResponse {
  text: string;
  detections: APIDetection[];
  processing_time?: number;
  model_used?: string;
}

// --- 아이콘 컴포넌트 ---
const UserIcon = () => (
  <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center text-white shrink-0">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
      <path fillRule="evenodd" d="M7.5 6a4.5 4.5 0 119 0 4.5 4.5 0 01-9 0zM3.751 20.105a8.25 8.25 0 0116.498 0 .75.75 0 01-.437.695A18.683 18.683 0 0112 22.5c-2.786 0-5.433-.608-7.812-1.7a.75.75 0 01-.437-.695z" clipRule="evenodd" />
    </svg>
  </div>
);

const BotIcon = () => (
  <div className="w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center text-white shrink-0">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
      <path fillRule="evenodd" d="M9.315 7.584C12.195 3.883 16.695 1.5 21.75 1.5a.75.75 0 01.75.75c0 5.056-2.383 9.555-6.084 12.436A6.75 6.75 0 019.75 22.5a.75.75 0 01-.75-.75v-4.131A15.838 15.838 0 016.382 15H2.25a.75.75 0 01-.75-.75 6.75 6.75 0 017.815-6.666zM15 6.75a2.25 2.25 0 100 4.5 2.25 2.25 0 000-4.5z" clipRule="evenodd" />
      <path d="M5.26 17.242a.75.75 0 10-.897-1.203 5.243 5.243 0 00-2.05 5.022.75.75 0 00.625.627 5.243 5.243 0 002.322-4.446z" />
    </svg>
  </div>
);

// --- 범례 아이템 컴포넌트 ---
const LegendItem = ({ color, label }: { color: string; label: string }) => (
  <div className="flex items-center space-x-2">
    <div className={`w-3 h-3 rounded-full ${color}`}></div>
    <span className="text-xs text-gray-400">{label}</span>
  </div>
);

export default function Page() {
  const [viewMode, setViewMode] = useState<'analyzer' | 'chat'>('analyzer');

  // 공통 상태
  const [userInput, setUserInput] = useState<string>('');
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [mobileTab, setMobileTab] = useState<'main' | 'history'>('main');

  // 분석기 상태
  const [analysisResult, setAnalysisResult] = useState<AnalysisSegment[]>([]);
  const [isAnalyzerLoading, setIsAnalyzerLoading] = useState<boolean>(false);
  const [analyzerError, setAnalyzerError] = useState<string | null>(null);

  // 채팅 상태
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<AnalysisSegment[]>([]);
  const [isChatStreaming, setIsChatStreaming] = useState<boolean>(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // 입력창 제어를 위한 Ref
  const chatInputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const savedHistory = localStorage.getItem('unifiedHistory');
    if (savedHistory) setHistory(JSON.parse(savedHistory));
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, streamingResponse]);

  const saveHistory = (text: string, result?: AnalysisSegment[]) => {
    const newLog: HistoryItem = {
      id: new Date().toISOString(),
      originalText: text,
      result: result,
      timestamp: new Date().toLocaleString('ko-KR', { hour: '2-digit', minute: '2-digit' }),
    };
    const updatedHistory = [newLog, ...history].slice(0, 50);
    setHistory(updatedHistory);
    localStorage.setItem('unifiedHistory', JSON.stringify(updatedHistory));
  };

  const getHighlightStyle = (type: AnalysisSegment['type'], confidence: number): string => {
    const baseStyle = 'px-1 rounded text-white font-semibold transition-colors duration-300';
    let colorStyle = '';
    switch (type) {
      case 'ORG': colorStyle = confidence >= 85 ? 'bg-blue-600' : 'bg-blue-500 bg-opacity-70'; break;
      case 'PER': colorStyle = confidence >= 85 ? 'bg-green-600' : 'bg-green-500 bg-opacity-70'; break;
      case 'LOC': colorStyle = confidence >= 85 ? 'bg-orange-500' : 'bg-orange-400 bg-opacity-70'; break;
      default: return '';
    }
    return `${baseStyle} ${colorStyle}`;
  };

  // API 호출 함수
  const analyzeTextAPI = async (text: string): Promise<AnalysisSegment[]> => {
    const response = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`API 오류: ${response.status}`);
    }

    const data: APIResponse = await response.json();

    // API 응답을 UI 세그먼트로 변환
    return convertAPIResponseToSegments(text, data.detections);
  };

  // API 응답을 UI 세그먼트로 변환하는 함수
  const convertAPIResponseToSegments = (originalText: string, detections: APIDetection[]): AnalysisSegment[] => {
    if (!detections || detections.length === 0) {
      return [{ text: originalText, type: 'normal', confidence: 100 }];
    }

    // 감지된 항목들을 시작 위치 순으로 정렬
    const sortedDetections = [...detections].sort((a, b) => a.start - b.start);

    const segments: AnalysisSegment[] = [];
    let lastEnd = 0;

    for (const detection of sortedDetections) {
      // 감지된 항목 이전의 일반 텍스트 추가
      if (detection.start > lastEnd) {
        segments.push({
          text: originalText.slice(lastEnd, detection.start),
          type: 'normal',
          confidence: 100,
        });
      }

      // 감지된 항목 추가 (타입 매핑)
      const entityType = mapEntityType(detection.type);
      segments.push({
        text: detection.entity || detection.text,
        type: entityType,
        confidence: Math.round(detection.confidence * 100),
      });

      lastEnd = detection.end;
    }

    // 마지막 감지 항목 이후의 텍스트 추가
    if (lastEnd < originalText.length) {
      segments.push({
        text: originalText.slice(lastEnd),
        type: 'normal',
        confidence: 100,
      });
    }

    return segments;
  };

  // 엔티티 타입 매핑
  const mapEntityType = (apiType: string): AnalysisSegment['type'] => {
    const typeMap: { [key: string]: AnalysisSegment['type'] } = {
      'company_anonymized': 'ORG',
      'person_initial': 'PER',
      'country_reference': 'LOC',
      'initial_company': 'ORG',
      'ORG': 'ORG',
      'PER': 'PER',
      'LOC': 'LOC',
      'ORGANIZATION': 'ORG',
      'PERSON': 'PER',
      'LOCATION': 'LOC',
    };
    return typeMap[apiType] || 'normal';
  };

  const handleAnalyzerRun = async () => {
    if (!userInput.trim()) return;

    setIsAnalyzerLoading(true);
    setAnalysisResult([]);
    setAnalyzerError(null);

    try {
      const response = await analyzeTextAPI(userInput);
      setAnalysisResult(response);
      saveHistory(userInput, response);
    } catch (error) {
      console.error('분석 오류:', error);
      setAnalyzerError(error instanceof Error ? error.message : '분석 중 오류가 발생했습니다.');
      // 에러 발생 시에도 기본 결과 표시
      setAnalysisResult([{ text: userInput, type: 'normal', confidence: 100 }]);
    } finally {
      setIsAnalyzerLoading(false);
    }
  };

  const handleChatSend = async () => {
    if (!userInput.trim() || isChatStreaming) return;

    const text = userInput;
    setUserInput('');
    setChatError(null);

    // 전송 직후 입력창에 강제로 포커스
    setTimeout(() => {
      chatInputRef.current?.focus();
    }, 0);

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    saveHistory(text);

    setIsChatStreaming(true);
    setStreamingResponse([]);

    try {
      const response = await analyzeTextAPI(text);

      // 스트리밍 효과 시뮬레이션
      for (let i = 0; i < response.length; i++) {
        await new Promise(resolve => setTimeout(resolve, 80));
        setStreamingResponse(prev => [...prev, response[i]]);
      }

      setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    } catch (error) {
      console.error('채팅 분석 오류:', error);
      setChatError(error instanceof Error ? error.message : '분석 중 오류가 발생했습니다.');
      const errorResponse: AnalysisSegment[] = [
        { text: '죄송합니다. 분석 중 오류가 발생했습니다.', type: 'normal', confidence: 100 }
      ];
      setMessages(prev => [...prev, { role: 'assistant', content: errorResponse }]);
    } finally {
      setStreamingResponse([]);
      setIsChatStreaming(false);

      // 스트리밍이 끝난 후에도 다시 포커스
      setTimeout(() => {
        chatInputRef.current?.focus();
      }, 0);
    }
  };

  const loadFromHistory = (item: HistoryItem) => {
    if (viewMode === 'analyzer') {
      setUserInput(item.originalText);
      if (item.result) setAnalysisResult(item.result);
    } else {
      setUserInput(item.originalText);
      setTimeout(() => chatInputRef.current?.focus(), 0);
    }
    setMobileTab('main');
  };

  const clearHistory = () => {
    setHistory([]);
    localStorage.removeItem('unifiedHistory');
  };

  const exportHistory = () => {
    const dataStr = JSON.stringify(history, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `euphemism-history-${new Date().toISOString().split('T')[0]}.json`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="bg-gray-950 h-screen flex flex-col items-center overflow-hidden font-sans">

      {/* 헤더 & 모드 스위처 */}
      <header className="w-full bg-gray-900 border-b border-gray-800 p-3 shrink-0 z-20">
        <div className="max-w-7xl mx-auto flex justify-between items-center">
             <h1 className="text-lg md:text-xl font-bold text-white flex items-center gap-2">
                <span className="bg-gradient-to-br from-blue-600 to-indigo-600 w-8 h-8 rounded-lg flex items-center justify-center text-xs shadow-lg">AI</span>
                Expression Detector
             </h1>

             <div className="flex bg-gray-800 p-1 rounded-lg">
                <button
                    onClick={() => setViewMode('analyzer')}
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                        viewMode === 'analyzer' ? 'bg-gray-700 text-white shadow' : 'text-gray-400 hover:text-gray-200'
                    }`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path d="M10 2a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 2zM10 15a.75.75 0 01.75.75v1.5a.75.75 0 01-1.5 0v-1.5A.75.75 0 0110 15zM10 7a3 3 0 100 6 3 3 0 000-6zM15.657 5.404a.75.75 0 10-1.06-1.06l-1.061 1.06a.75.75 0 001.06 1.06l1.06-1.06zM6.464 14.596a.75.75 0 10-1.06-1.06l-1.06 1.06a.75.75 0 001.06 1.06l1.06-1.06zM18 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 0118 10zM5 10a.75.75 0 01-.75.75h-1.5a.75.75 0 010-1.5h1.5A.75.75 0 015 10zM14.596 15.657a.75.75 0 001.06-1.06l-1.06-1.061a.75.75 0 10-1.06 1.06l1.06 1.06zM5.404 6.464a.75.75 0 001.06-1.06l-1.06-1.06a.75.75 0 10-1.061 1.06l1.06 1.06z" /></svg>
                    분석기
                </button>
                <button
                    onClick={() => {
                        setViewMode('chat');
                        setTimeout(() => chatInputRef.current?.focus(), 100);
                    }}
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-md text-sm font-medium transition-all ${
                        viewMode === 'chat' ? 'bg-blue-600 text-white shadow' : 'text-gray-400 hover:text-gray-200'
                    }`}
                >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-4 h-4"><path fillRule="evenodd" d="M10 2c-2.236 0-4.43.18-6.57.524C1.993 2.755 1 4.014 1 5.426v5.148c0 1.413.993 2.67 2.43 2.902.848.137 1.705.248 2.57.331v3.443a.75.75 0 001.28.53l3.58-3.579a.78.78 0 01.527-.224 41.202 41.202 0 003.444-.33c1.436-.23 2.429-1.487 2.429-2.902V5.426c0-1.413-.993-2.67-2.43-2.902A41.289 41.289 0 0010 2zm0 7a1 1 0 100-2 1 1 0 000 2zM8 8a1 1 0 11-2 0 1 1 0 012 0zm5 1a1 1 0 100-2 1 1 0 000 2z" clipRule="evenodd" /></svg>
                    채팅
                </button>
             </div>

             <button className="lg:hidden text-gray-400" onClick={() => setMobileTab(mobileTab === 'main' ? 'history' : 'main')}>
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-6 h-6"><path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" /></svg>
             </button>
        </div>
      </header>

      <div className="flex w-full max-w-7xl flex-grow min-h-0 relative">

        {viewMode === 'analyzer' && (
            <div className={`w-full h-full p-4 lg:p-6 gap-6 grid grid-cols-1 lg:grid-cols-2 transition-opacity duration-300 ${mobileTab === 'main' ? 'block' : 'hidden lg:grid'}`}>
                {/* 원문 입력 패널 */}
                <div className="flex flex-col h-full bg-gray-900 rounded-2xl border border-gray-800 shadow-xl overflow-hidden">
                    <div className="bg-gray-800 px-4 py-3 border-b border-gray-700 flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300 flex items-center gap-2">[EDIT] 원문 입력</span>
                        <button onClick={() => setUserInput('')} className="text-xs text-gray-500 hover:text-red-400">지우기</button>
                    </div>
                    <textarea
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        className="flex-grow w-full p-4 bg-transparent text-gray-200 resize-none focus:outline-none text-base leading-relaxed"
                        placeholder="분석할 텍스트를 여기에 붙여넣거나 입력하세요..."
                    />
                    <div className="p-4 border-t border-gray-800 bg-gray-900/50">
                        <button onClick={handleAnalyzerRun} disabled={isAnalyzerLoading || !userInput.trim()} className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition shadow-md flex justify-center items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed">
                            {isAnalyzerLoading ? (
                              <>
                                <span className="animate-spin">⟳</span>
                                <span>분석 중...</span>
                              </>
                            ) : (
                              <span>[SEARCH] 분석 실행</span>
                            )}
                        </button>
                    </div>
                </div>

                {/* 분석 결과 패널 */}
                <div className="flex flex-col h-full bg-gray-900 rounded-2xl border border-gray-800 shadow-xl overflow-hidden relative">
                    <div className="bg-gray-800 px-4 py-3 border-b border-gray-700 flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300 flex items-center gap-2">[STATS] 분석 결과</span>
                        <div className="flex gap-3">
                          <LegendItem color="bg-blue-600" label="조직" />
                          <LegendItem color="bg-green-600" label="인물" />
                          <LegendItem color="bg-orange-500" label="장소" />
                        </div>
                    </div>
                    <div className="flex-grow w-full p-6 overflow-y-auto leading-loose text-lg text-gray-300">
                        {analyzerError && (
                          <div className="mb-4 p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm">
                            [WARNING] {analyzerError}
                          </div>
                        )}
                        {analysisResult.length > 0 ? (
                          analysisResult.map((segment, index) => (
                            <span key={index} className={getHighlightStyle(segment.type, segment.confidence)}>
                              {segment.text}
                            </span>
                          ))
                        ) : (
                          <div className="h-full flex flex-col items-center justify-center text-gray-600 gap-3">
                            <p>왼쪽에서 텍스트를 입력하고 분석을 시작하세요.</p>
                          </div>
                        )}
                    </div>
                </div>
            </div>
        )}

        {viewMode === 'chat' && (
            <div className={`flex flex-col h-full w-full bg-gray-950 transition-opacity duration-300 ${mobileTab === 'main' ? 'flex' : 'hidden lg:flex'}`}>
                <div ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 space-y-6 scroll-smooth">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-gray-600 space-y-3 opacity-60">
                            <BotIcon />
                            <p className="text-sm">무엇이든 물어보세요! 실시간으로 분석해드립니다.</p>
                        </div>
                    )}
                    {chatError && (
                      <div className="mx-auto max-w-md p-3 bg-red-900/30 border border-red-700 rounded-lg text-red-400 text-sm text-center">
                        [WARNING] {chatError}
                      </div>
                    )}
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'assistant' && <BotIcon />}
                            <div className={`max-w-[75%] p-3.5 rounded-2xl shadow-sm text-sm md:text-base leading-relaxed ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-gray-800 text-gray-200 rounded-tl-none border border-gray-700'}`}>
                                {msg.role === 'user' ? (
                                  <p className="whitespace-pre-wrap">{msg.content as string}</p>
                                ) : (
                                  <div>
                                    {(msg.content as AnalysisSegment[]).map((seg, i) => (
                                      <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>
                                    ))}
                                  </div>
                                )}
                            </div>
                            {msg.role === 'user' && <UserIcon />}
                        </div>
                    ))}
                    {isChatStreaming && (
                        <div className="flex gap-3 justify-start">
                          <BotIcon />
                          <div className="max-w-[75%] p-4 rounded-2xl rounded-tl-none bg-gray-800 text-gray-200 border border-gray-700 shadow-sm">
                            {streamingResponse.map((seg, i) => (
                              <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>
                            ))}
                            <span className="inline-block w-1.5 h-4 ml-1 bg-indigo-500 animate-pulse rounded-full align-middle"></span>
                          </div>
                        </div>
                    )}
                </div>

                <div className="p-4 bg-gray-900 border-t border-gray-800">
                    <div className="max-w-4xl mx-auto flex items-end gap-2 bg-gray-800 p-2 rounded-2xl border border-gray-700 focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500 transition-all shadow-lg">
                        <textarea
                          ref={chatInputRef}
                          value={userInput}
                          onChange={(e) => setUserInput(e.target.value)}
                          onKeyDown={(e) => {
                              // 한글 조합 중이면 전송하지 않음
                              if (e.nativeEvent.isComposing) return;

                              if (e.key === 'Enter' && !e.shiftKey) {
                                  e.preventDefault();
                                  handleChatSend();
                              }
                          }}
                          placeholder="메시지를 입력하세요..."
                          className="w-full bg-transparent text-white p-2.5 max-h-32 min-h-[44px] resize-none focus:outline-none"
                          rows={1}
                      />
                        <button
                            onClick={handleChatSend}
                            disabled={!userInput.trim() || isChatStreaming}
                            className="p-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-xl shrink-0 transition-colors"
                        >
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5"><path d="M3.478 2.405a.75.75 0 00-.926.94l2.432 7.905H13.5a.75.75 0 010 1.5H4.984l-2.432 7.905a.75.75 0 00.926.94 60.519 60.519 0 0018.445-8.986.75.75 0 000-1.218A60.517 60.517 0 003.478 2.405z" /></svg>
                        </button>
                    </div>
                </div>
            </div>
        )}

        {/* 히스토리 사이드바 */}
        <div className={`flex-col gap-4 border-l border-gray-800 bg-gray-900 p-4 w-full lg:w-80 lg:flex lg:static lg:h-auto ${mobileTab === 'history' ? 'absolute inset-0 z-20 flex' : 'hidden'}`}>
            <div className="bg-gray-800 rounded-xl shadow-lg p-5 flex flex-col flex-grow min-h-0 border border-gray-700">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-gray-700">
                    <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      History
                    </h3>
                    <div className="flex gap-2">
                      <button
                        title="내보내기"
                        onClick={exportHistory}
                        className="text-gray-500 hover:text-white transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
                        </svg>
                      </button>
                      <button
                        title="전체 삭제"
                        onClick={clearHistory}
                        className="text-gray-500 hover:text-red-400 transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                        </svg>
                      </button>
                    </div>
                </div>
                <div className="overflow-y-auto flex-grow pr-1 space-y-2">
                    {history.length > 0 ? (
                      history.map((item) => (
                        <button
                          key={item.id}
                          onClick={() => loadFromHistory(item)}
                          className="w-full text-left p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition group border border-transparent hover:border-gray-600"
                        >
                          <p className="text-sm text-gray-200 truncate group-hover:text-white font-medium">{item.originalText}</p>
                          <p className="text-xs text-gray-500 mt-1 flex justify-between">
                            <span>{item.timestamp}</span>
                            {item.result && <span className="text-[10px] bg-gray-600 px-1 rounded text-gray-300">분석</span>}
                          </p>
                        </button>
                      ))
                    ) : (
                      <div className="text-gray-600 text-xs text-center py-10 flex flex-col items-center">
                        <span>기록이 없습니다.</span>
                      </div>
                    )}
                </div>
            </div>
        </div>

      </div>
    </div>
  );
}
