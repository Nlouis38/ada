import ollama
import asyncio
import websockets
import json
import base64
import pyaudio
from RealtimeSTT import AudioToTextRecorder
import torch  # Import the torch library
import re
import time
from WIDGETS import system, timer, project

ELEVENLABS_API_KEY = 'sk_da39fd7f9dae2fbd9bb71960b6bcffda1d57cd7249c636d4'
VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

class ADA:
    def __init__(self):
        print("initializing...")

        # Check for CUDA availability
        if torch.cuda.is_available():
            self.device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            self.device = "cpu"
            print("CUDA is not available. Using CPU.")

        self.model = "gemma3:4b-it-q4_K_M"
        self.system_behavior = """
            Your name is ADA (Advanced Design Assistant) you are a helpful AI assistant.  You are an expert in All STEM Fields providing concise and accurate information. When asked to perform a task, respond with the code to perform that task wrapped in ```tool_code```.  If the task does not require a function call, provide a direct answer without using ```tool_code```.  Always respond in a helpful and informative manner."

            You speak with a british accent and address people as Sir.
        """

        self.instruction_prompt_with_function_calling = '''
            At each turn, if you decide to invoke any of the function(s), it should be wrapped with ```tool_code```. If you decide to call a function the response should only have the function wrapped in tool code nothing more. The python methods described below are imported and available, you can only use defined methods also only call methods when you are sure they need to be called. The generated code should be readable and efficient. 
            
            The response to a method will be wrapped in ```tool_output``` use the response to give the user an answer based on the information provided that is wrapped in ```tool_ouput```.

            For regular prompts do not call any functions or wrap the response in ```tool_code```.

            The following Python methods are available:

            ```python
            def camera.open() -> None:
                """Open the camera"""

            def system.info() -> None:
                """ Gathers and prints system information including CPU, RAM, and GPU details. Only call when user ask about computer information. """

            def timer.set(time_str):
                """
                Counts down from a specified time in HH:MM:SS format.

                Args:
                    time_str (str): The time to count down from in HH:MM:SS format.
                """
            def project.create_folder(folder_name):
                """
                Creates a project folder and a text file to store chat history.

                Args:
                    folder_name (str): The name of the project folder to create.
                """
        ```

        User: {user_message}
        '''

        self.model_params = {
            'temperature': 0.1,
            'top_p': 0.9,
        }
        self.conversation_history = []

        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue()
        self.recorder_config = {
            'model': 'large-v3',
            'spinner': False,
            'language': 'en',
            'silero_sensitivity': 0.01,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': 0.1,
            'min_length_of_recording': 0.2,
            'min_gap_between_recordings': 0,

            #'realtime_model_type': 'tiny.en',
            #'enable_realtime_transcription': True,
            #'on_realtime_transcription_update': self.clear_queues,
        }

        try:
            self.recorder = AudioToTextRecorder(**self.recorder_config)
        except Exception as e:
            print(f"Error initializing AudioToTextRecorder: {e}")
            self.recorder = None  # Or handle this appropriately

        try:
            self.pya = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            self.pya = None
        
        self.response_start_time = None
        self.audio_start_time = None

    async def clear_queues(self, text=""):
        """Clears all data from the input, response, and audio queues."""
        queues = [self.input_queue, self.response_queue, self.audio_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break  # Queue is empty

    async def input_message(self):
        while True:
            try:
                prompt = await asyncio.to_thread(input, "Enter your message: ")
                if prompt.lower() == "exit":
                    await self.input_queue.put(None)  # Signal to exit
                    break
                await self.clear_queues()
                await self.input_queue.put(prompt)
            except Exception as e:
                print(f"Error in input_message: {e}")
                continue  # Continue the loop even if there's an error

    async def send_prompt(self):
        while True:
            try:
                prompt = await self.input_queue.get()
                if prompt is None:
                    break  # Exit loop if None is received
                
                self.response_start_time = time.time() #start timer when prompt is sent
                
                messages = [{"role": "system", "content": self.system_behavior}] + self.conversation_history + [{"role": "user", "content": self.instruction_prompt_with_function_calling.format(user_message=prompt)}]
                try:
                    response = ollama.chat(model=self.model, messages=messages, stream=True)
                    full_response = ""
                    in_function_call = False
                    function_call = ""

                    for chunk in response:
                        chunk_content = chunk['message']['content']
                        if chunk_content == "```":
                            if in_function_call == True:
                                in_function_call = False
                                function_call += "```"
                                tool_output = self.extract_tool_call(function_call)

                                messages = [{"role": "system", "content": self.system_behavior}] + self.conversation_history + [{"role": "user", "content": self.instruction_prompt_with_function_calling.format(user_message=tool_output)}]
                                
                                response = ollama.chat(model=self.model, messages=messages, stream=True)
                                for chunk in response:
                                    chunk_content = chunk['message']['content']
                                    print(chunk_content, end="", flush=True)
                                    await self.response_queue.put(chunk_content)
                                print()
                                continue
                            else:
                               in_function_call = True

                        if in_function_call == False:
                            await self.response_queue.put(chunk_content)
                            await asyncio.sleep(0)
                        else:
                            function_call += chunk_content                        
                        if chunk_content:
                            print(chunk_content, end="", flush=True) #print chunks on same line
                            full_response += chunk_content
                    print() # new line
                    self.conversation_history.append({"role": "user", "content": prompt})
                    self.conversation_history.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    print(f"An error occurred in send_prompt: {e}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Unexpected error in send_prompt: {e}")

            finally:  # Ensure the sentinel value is added even if an error occurs
                await self.response_queue.put(None)

    def extract_tool_call(self, text):
        import io
        from contextlib import redirect_stdout

        pattern = r"```tool_code\s*(.*?)\s*```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            # Capture stdout in a string buffer
            f = io.StringIO()
            with redirect_stdout(f):
                result = eval(code)
            output = f.getvalue()
            r = result if output == '' else output
            return f'```tool_output\n{str(r).strip()}\n```'''
        return None

    async def tts(self):
        """Send text to ElevenLabs API and stream the returned audio."""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    try:
                        await websocket.send(json.dumps({
                            "text": " ",
                            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "speed": 1.1},
                            "xi_api_key": ELEVENLABS_API_KEY,
                        }))

                        async def listen():
                            """Listen to the websocket for audio data and stream it."""
                            while True:
                                try:
                                    message = await websocket.recv()
                                    data = json.loads(message)
                                    print(data.get("normalizedAlignment"))
                                    if data.get("audio"):
                                        if self.audio_start_time is None:
                                            self.audio_start_time = time.time()
                                            if self.response_start_time is not None:
                                                latency = self.audio_start_time - self.response_start_time
                                                print(f"Time from prompt to first audio byte: {latency:.4f} seconds")
                                        await self.audio_queue.put(base64.b64decode(data["audio"]))
                                    elif data.get('isFinal'):
                                        break  # Exit listen loop when isFinal is received
                                except websockets.exceptions.ConnectionClosed:
                                    print("Connection closed in listen")
                                    break
                                except json.JSONDecodeError as e:
                                    print(f"JSON Decode Error in listen: {e}")
                                    break
                                except Exception as e:
                                    print(f"Error in listen: {e}")
                                    break

                        listen_task = asyncio.create_task(listen())

                        try:
                            while True:
                                text = await self.response_queue.get()
                                if text is None:
                                    break
                                await websocket.send(json.dumps({"text": text}))
                                print(json.dumps({"text": text}))
                        except Exception as e:
                            print(f"Error processing text: {e}")
                        finally:
                            await listen_task  # Ensure listen_task completes
                            self.response_start_time = None
                            self.audio_start_time = None

                    except websockets.exceptions.WebSocketException as e:
                        print(f"WebSocket error: {e}")
                    except Exception as e:
                        print(f"Error during websocket communication: {e}")

            except websockets.exceptions.WebSocketException as e:
                print(f"WebSocket connection error: {e}")
            except Exception as e:
                print(f"Error connecting to websocket: {e}")

    async def play_audio(self):
        if self.pya is None:
            print("PyAudio is not initialized.")
            return

        try:
            stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            while True:
                try:
                    bytestream = await self.audio_queue.get()
                    await asyncio.to_thread(stream.write, bytestream)
                except asyncio.CancelledError:
                    break  # Exit loop if task is cancelled
                except Exception as e:
                    print(f"Error in play_audio loop: {e}")
        except pyaudio.PyAudioError as e:
            print(f"PyAudio error: {e}")
        except Exception as e:
            print(f"Error opening audio stream: {e}")

    async def stt(self):
        if self.recorder is None:
            print("Audio recorder is not initialized.")
            return

        while True:
            try:
                text = await asyncio.to_thread(self.recorder.text)
                await self.clear_queues()
                await self.input_queue.put(text)
                print(text)
            except Exception as e:
                print(f"Error in listen: {e}")
                continue  # Continue the loop even if there's an error
