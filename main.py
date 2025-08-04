# /home/ubuntu/unified-inference-service/main.py

import threading
import time
import queue
import asyncio
import os
from dataclasses import dataclass, field
from concurrent.futures import Future
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Load Configuration from .env ---
load_dotenv()
MODEL_NAME = os.getenv("MODEL_NAME", "mistralai/Mistral-Small-3.1-24B-Instruct-2503")

# --- Global event to signal model readiness ---
model_ready_event = threading.Event()

# --- Data Structures ---
@dataclass(order=True)
class PriorityRequest:
    priority: int
    prompt_data: Dict[str, Any] = field(compare=False)
    future: Future = field(compare=False)
    agent_id: str = field(compare=False)
    task_type: str = field(compare=False)

# --- Model Worker ---
class ModelWorker(threading.Thread):
    def __init__(self, request_queue: queue.PriorityQueue):
        super().__init__()
        self.request_queue = request_queue
        self.llm = None
        self.daemon = True

    def load_model(self):
        print(f"[{self.name}] Initializing vLLM engine for model: {MODEL_NAME}...")
        try:
            from vllm import LLM
            import torch
            
            # Determine tensor parallel size dynamically based on available GPUs
            gpu_count = torch.cuda.device_count()
            effective_tp_size = gpu_count if gpu_count > 0 else 1
            print(f"[{self.name}] Detected {gpu_count} GPU(s). Using tensor_parallel_size={effective_tp_size}.")

            self.llm = LLM(
                model=MODEL_NAME,
                tokenizer=MODEL_NAME,
                trust_remote_code=True,
                dtype="float16",  
                tensor_parallel_size=effective_tp_size,
                gpu_memory_utilization=0.90,
                max_model_len=65000,  
                # quantization="bitsandbytes",
            )
            
            print(f"[{self.name}] vLLM engine loaded successfully.")
            model_ready_event.set()
        except Exception as e:
            print(f"[{self.name}] FATAL: Failed to load vLLM engine. Error: {e}")
            model_ready_event.set()
            raise

    def run(self):
        self.load_model()

        while True:
            try:
                request = self.request_queue.get()
                if request is None:
                    break

                print(f"[{self.name}] Handling request: agent={request.agent_id}, type={request.task_type}, prio={request.priority}")
                start = time.time()

                messages = request.prompt_data["messages"]
                sampling_params = request.prompt_data["sampling_params"]

                # prompt_as_string = "".join([msg.get("content", "") for msg in messages])
                
                # print(f"[{self.name}] Received final prompt (len={len(prompt_as_string)}): {prompt_as_string}...")

                # Perform inference on the client-provided prompt string
                outputs = self.llm.chat(messages, sampling_params=sampling_params)
                result_text = outputs[0].outputs[0].text.strip()

                print(f"[{self.name}] Response: {result_text}...")  
                request.future.set_result(result_text)

                print(f"[{self.name}] Completed in {time.time() - start:.2f}s")

            except Exception as e:
                print(f"[{self.name}] ERROR: {e}")
                if 'request' in locals() and isinstance(request, PriorityRequest):
                    request.future.set_exception(e)
            finally:
                if 'request' in locals() and request is not None:
                    self.request_queue.task_done()

# --- Inference Manager ---
class InferenceManager:
    def __init__(self):
        self.request_queue = queue.PriorityQueue()
        self.worker = ModelWorker(self.request_queue)
        self.worker.start()

    def submit_request(self, prompt_data: Dict, priority: int, agent_id: str, task_type: str) -> Future:
        future = Future()
        request = PriorityRequest(
            priority=priority,
            prompt_data=prompt_data,
            future=future,
            agent_id=agent_id,
            task_type=task_type,
        )
        self.request_queue.put(request)
        return future

    def shutdown(self):
        self.request_queue.put(None)
        self.worker.join()

# --- API Models ---
class InvokePayload(BaseModel):
    agent_id: str
    task_type: str
    priority: int
    messages: List[Dict[str, str]]
    sampling_params: Dict[str, Any]

# --- FastAPI App ---
app = FastAPI(title="Unified LLM Inference Service")
manager: InferenceManager | None = None

@app.on_event("startup")
def startup_event():
    global manager
    print("[API] Starting inference manager...")
    manager = InferenceManager()

@app.on_event("shutdown")
def shutdown_event():
    global manager
    if manager:
        print("[API] Shutting down inference manager...")
        manager.shutdown()

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not model_ready_event.is_set():
        raise HTTPException(status_code=503, detail="Model is still loading")
    
    if not manager or not manager.worker.is_alive():
        raise HTTPException(status_code=503, detail="Worker thread has died. Service is unhealthy.")

    if manager.worker.llm is None:
        raise HTTPException(status_code=503, detail="Model failed to initialize.")
    
    return {"status": "ready"}

@app.post("/invoke", response_model=Dict[str, Any])
async def invoke_model(payload: InvokePayload):
    if not model_ready_event.is_set() or not manager or not manager.worker.llm:
        raise HTTPException(status_code=503, detail="Model not ready")

    from vllm import SamplingParams
    sampling_params_obj = SamplingParams(**payload.sampling_params)
    prompt_data = {"messages": payload.messages, "sampling_params": sampling_params_obj}

    future = manager.submit_request(prompt_data, payload.priority, payload.agent_id, payload.task_type)
    result = await asyncio.wrap_future(future)
    return {"response": result}
