import json
import pathlib
from typing import List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent_runner import run_research_pipeline
from agent.configuration import Configuration

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class MessageInput(BaseModel):
    type: str
    content: str
    id: str = ""


class ResearchRequest(BaseModel):
    messages: List[MessageInput]
    initial_search_query_count: int = 3
    max_research_loops: int = 2
    reasoning_model: str = "gemini-2.5-pro"


# ---------------------------------------------------------------------------
# SSE research endpoint
# ---------------------------------------------------------------------------

@app.post("/api/research/stream")
async def research_stream(request: ResearchRequest):
    messages = [m.model_dump() for m in request.messages]
    config = Configuration.from_config()

    async def event_generator():
        try:
            async for event in run_research_pipeline(
                messages=messages,
                initial_search_query_count=request.initial_search_query_count,
                max_research_loops=request.max_research_loops,
                reasoning_model=request.reasoning_model,
                config=config,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Frontend static file serving
# ---------------------------------------------------------------------------

def create_frontend_router(build_dir="../frontend/dist"):
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
