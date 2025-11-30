"""
WebSocket routes for real-time analysis
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
from typing import Set

from app.ml.detector import EuphemismDetector

router = APIRouter(prefix="/ws", tags=["websocket"])

# Global detector instance
detector = EuphemismDetector()

# Active connections
active_connections: Set[WebSocket] = set()


@router.websocket("/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    Real-time euphemism detection via WebSocket
    
    Client sends JSON: {"text": ""}
    Server responds with analysis results
    """
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse JSON
                message = json.loads(data)
                text = message.get('text', '')
                
                if not text:
                    await websocket.send_json({
                        'error': 'Text cannot be empty'
                    })
                    continue
                
                # Analyze text
                result = await detector.stream_analyze(text)
                
                # Send result
                await websocket.send_json(result)
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    'error': 'Invalid JSON format'
                })
            except Exception as e:
                await websocket.send_json({
                    'error': f'Analysis failed: {str(e)}'
                })
    
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


@router.get("/connections")
async def get_connection_count():
    """Get number of active WebSocket connections"""
    return {
        "active_connections": len(active_connections)
    }
