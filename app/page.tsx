'use client';

import React, { useState, useEffect, useRef } from 'react';

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

  // 채팅 상태
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingResponse, setStreamingResponse] = useState<AnalysisSegment[]>([]);
  const [isChatStreaming, setIsChatStreaming] = useState<boolean>(false);
  const chatContainerRef = useRef<HTMLDivElement>(null);
  
  // ★★★ 1. 입력창 제어를 위한 Ref 추가 ★★★
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
    let baseStyle = 'px-1 rounded text-white font-semibold transition-colors duration-300';
    let colorStyle = '';
    switch (type) {
      case 'ORG': colorStyle = confidence >= 85 ? 'bg-blue-600' : 'bg-blue-500 bg-opacity-70'; break;
      case 'PER': colorStyle = confidence >= 85 ? 'bg-green-600' : 'bg-green-500 bg-opacity-70'; break;
      case 'LOC': colorStyle = confidence >= 85 ? 'bg-orange-500' : 'bg-orange-400 bg-opacity-70'; break;
      default: return '';
    }
    return `${baseStyle} ${colorStyle}`;
  };

  const generateMockResponse = (text: string): AnalysisSegment[] => {
    return [
      { text: '입력하신 내용에서 ', type: 'normal', confidence: 100 },
      { text: '삼성전자', type: 'ORG', confidence: 95 },
      { text: '와 ', type: 'normal', confidence: 100 },
      { text: '이재용', type: 'PER', confidence: 88 },
      { text: ' 회장을 탐지했습니다.', type: 'normal', confidence: 100 },
    ];
  };

  const handleAnalyzerRun = async () => {
    if (!userInput.trim()) return;
    setIsAnalyzerLoading(true);
    setAnalysisResult([]);
    const response = generateMockResponse(userInput);
    await new Promise(resolve => setTimeout(resolve, 800)); 
    setAnalysisResult(response);
    saveHistory(userInput, response);
    setIsAnalyzerLoading(false);
  };

  const handleChatSend = async () => {
    if (!userInput.trim() || isChatStreaming) return;
    const text = userInput;
    setUserInput('');
    
    // ★★★ 2. 전송 직후 입력창에 강제로 포커스 주기 ★★★
    // setTimeout을 0으로 주면 렌더링 직후에 실행되어 더 안정적입니다.
    setTimeout(() => {
        chatInputRef.current?.focus();
    }, 0);

    setMessages(prev => [...prev, { role: 'user', content: text }]);
    saveHistory(text);

    setIsChatStreaming(true);
    setStreamingResponse([]);
    const response = generateMockResponse(text);

    for (let i = 0; i < response.length; i++) {
      await new Promise(resolve => setTimeout(resolve, 80));
      setStreamingResponse(prev => [...prev, response[i]]);
    }

    setMessages(prev => [...prev, { role: 'assistant', content: response }]);
    setStreamingResponse([]);
    setIsChatStreaming(false);
    
    // 스트리밍이 끝난 후에도 다시 포커스 (선택 사항)
    setTimeout(() => {
        chatInputRef.current?.focus();
    }, 0);
  };

  const loadFromHistory = (item: HistoryItem) => {
    if (viewMode === 'analyzer') {
      setUserInput(item.originalText);
      if (item.result) setAnalysisResult(item.result);
    } else {
      setUserInput(item.originalText);
      // 기록에서 불러왔을 때도 입력창에 포커스
      setTimeout(() => chatInputRef.current?.focus(), 0);
    }
    setMobileTab('main');
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
                        setTimeout(() => chatInputRef.current?.focus(), 100); // 모드 변경 시에도 포커스
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
                {/* 분석기 모드 코드 (이전과 동일) */}
                <div className="flex flex-col h-full bg-gray-900 rounded-2xl border border-gray-800 shadow-xl overflow-hidden">
                    <div className="bg-gray-800 px-4 py-3 border-b border-gray-700 flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300 flex items-center gap-2">✏️ 원문 입력</span>
                        <button onClick={() => setUserInput('')} className="text-xs text-gray-500 hover:text-red-400">지우기</button>
                    </div>
                    <textarea
                        value={userInput}
                        onChange={(e) => setUserInput(e.target.value)}
                        className="flex-grow w-full p-4 bg-transparent text-gray-200 resize-none focus:outline-none text-base leading-relaxed"
                        placeholder="분석할 텍스트를 여기에 붙여넣거나 입력하세요..."
                    />
                    <div className="p-4 border-t border-gray-800 bg-gray-900/50">
                        <button onClick={handleAnalyzerRun} disabled={isAnalyzerLoading || !userInput.trim()} className="w-full py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg transition shadow-md flex justify-center items-center gap-2">
                            {isAnalyzerLoading ? <span className="animate-spin text-xl">⟳</span> : <span>🔍 분석 실행</span>}
                        </button>
                    </div>
                </div>

                <div className="flex flex-col h-full bg-gray-900 rounded-2xl border border-gray-800 shadow-xl overflow-hidden relative">
                    <div className="bg-gray-800 px-4 py-3 border-b border-gray-700 flex justify-between items-center">
                        <span className="text-sm font-semibold text-gray-300 flex items-center gap-2">📊 분석 결과</span>
                        <div className="flex gap-3"><LegendItem color="bg-blue-600" label="조직" /><LegendItem color="bg-green-600" label="인물" /></div>
                    </div>
                    <div className="flex-grow w-full p-6 overflow-y-auto leading-loose text-lg text-gray-300">
                        {analysisResult.length > 0 ? (analysisResult.map((segment, index) => <span key={index} className={getHighlightStyle(segment.type, segment.confidence)}>{segment.text}</span>)) : (<div className="h-full flex flex-col items-center justify-center text-gray-600 gap-3"><p>왼쪽에서 텍스트를 입력하고 분석을 시작하세요.</p></div>)}
                    </div>
                </div>
            </div>
        )}

        {viewMode === 'chat' && (
            <div className={`flex flex-col h-full w-full bg-gray-950 transition-opacity duration-300 ${mobileTab === 'main' ? 'flex' : 'hidden lg:flex'}`}>
                <div ref={chatContainerRef} className="flex-grow overflow-y-auto p-4 space-y-6 scroll-smooth">
                    {messages.length === 0 && (
                        <div className="flex flex-col items-center justify-center h-full text-gray-600 space-y-3 opacity-60">
                            <BotIcon /><p className="text-sm">무엇이든 물어보세요! 실시간으로 분석해드립니다.</p>
                        </div>
                    )}
                    {messages.map((msg, idx) => (
                        <div key={idx} className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            {msg.role === 'assistant' && <BotIcon />}
                            <div className={`max-w-[75%] p-3.5 rounded-2xl shadow-sm text-sm md:text-base leading-relaxed ${msg.role === 'user' ? 'bg-indigo-600 text-white rounded-tr-none' : 'bg-gray-800 text-gray-200 rounded-tl-none border border-gray-700'}`}>
                                {msg.role === 'user' ? <p className="whitespace-pre-wrap">{msg.content as string}</p> : <div>{(msg.content as AnalysisSegment[]).map((seg, i) => <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>)}</div>}
                            </div>
                            {msg.role === 'user' && <UserIcon />}
                        </div>
                    ))}
                    {isChatStreaming && (
                        <div className="flex gap-3 justify-start"><BotIcon /><div className="max-w-[75%] p-4 rounded-2xl rounded-tl-none bg-gray-800 text-gray-200 border border-gray-700 shadow-sm">{streamingResponse.map((seg, i) => <span key={i} className={getHighlightStyle(seg.type, seg.confidence)}>{seg.text}</span>)}<span className="inline-block w-1.5 h-4 ml-1 bg-indigo-500 animate-pulse rounded-full align-middle"></span></div></div>
                    )}
                </div>
                
                <div className="p-4 bg-gray-900 border-t border-gray-800">
                    <div className="max-w-4xl mx-auto flex items-end gap-2 bg-gray-800 p-2 rounded-2xl border border-gray-700 focus-within:border-indigo-500 focus-within:ring-1 focus-within:ring-indigo-500 transition-all shadow-lg">
                        <textarea
                          ref={chatInputRef}
                          value={userInput}
                          onChange={(e) => setUserInput(e.target.value)}
                          
                          // ★★★ 여기가 수정된 핵심 부분입니다 ★★★
                          onKeyDown={(e) => {
                              // 1. 한글이 아직 조합 중(밑줄)이라면 전송하지 않고 그냥 리턴합니다.
                              // 이렇게 하면 엔터키가 '전송'이 아니라 '글자 확정' 역할을 하게 됩니다.
                              if (e.nativeEvent.isComposing) return;
                              
                              // 2. 조합이 끝난 상태에서 엔터를 눌렀을 때만 전송합니다.
                              if (e.key === 'Enter' && !e.shiftKey) {
                                  e.preventDefault();
                                  handleChatSend();
                              }
                          }}
                          // ★★★ ------------------------- ★★★
                          
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

        <div className={`flex-col gap-4 border-l border-gray-800 bg-gray-900 p-4 w-full lg:w-80 lg:flex lg:static lg:h-auto ${mobileTab === 'history' ? 'absolute inset-0 z-20 flex' : 'hidden'}`}>
            <div className="bg-gray-800 rounded-xl shadow-lg p-5 flex flex-col flex-grow min-h-0 border border-gray-700">
                <div className="flex justify-between items-center mb-4 pb-2 border-b border-gray-700">
                    <h3 className="text-sm font-bold text-gray-300 uppercase tracking-wider flex items-center gap-2"><svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" /></svg> History</h3>
                    <button title="내보내기" onClick={() => alert('백엔드 연동이 필요합니다.')} className="text-gray-500 hover:text-white transition-colors"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor" className="w-5 h-5"><path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" /></svg></button>
                </div>
                <div className="overflow-y-auto flex-grow pr-1 space-y-2">
                    {history.length > 0 ? (history.map((item) => <button key={item.id} onClick={() => loadFromHistory(item)} className="w-full text-left p-3 bg-gray-700/50 hover:bg-gray-700 rounded-lg transition group border border-transparent hover:border-gray-600"><p className="text-sm text-gray-200 truncate group-hover:text-white font-medium">{item.originalText}</p><p className="text-xs text-gray-500 mt-1 flex justify-between"><span>{item.timestamp}</span>{item.result && <span className="text-[10px] bg-gray-600 px-1 rounded text-gray-300">분석</span>}</p></button>)) : (<div className="text-gray-600 text-xs text-center py-10 flex flex-col items-center"><span>기록이 없습니다.</span></div>)}
                </div>
            </div>
        </div>

      </div>
    </div>
  );
}