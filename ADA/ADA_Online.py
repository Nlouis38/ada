import asyncio
import websockets
import json
import base64
import pyaudio
from RealtimeSTT import AudioToTextRecorder
import torch  # Import the torch library
import re
from google.genai import types
import asyncio
from google import genai
import os
from google.genai.types import Tool, GoogleSearch, Part, Blob, Content
import python_weather
import googlemaps # Added for travel duration
from datetime import datetime # Added for travel duration
from dotenv import load_dotenv # Added for API key loading

# --- Load Environment Variables ---
load_dotenv()
ELEVENLABS_API_KEY = os.getenv(sk_f978538247677a56d4851e6603367e805df0563233500740)
GOOGLE_API_KEY = os.getenv(AIzaSyD9KxpxCr0_9JQv8Osl_Y8aK1uJX9TMcZk)
MAPS_API_KEY = os.getenv(AIzaSyD9KxpxCr0_9JQv8Osl_Y8aK1uJX9TMcZk) # Added Maps API Key

# --- Validate API Keys ---
if not ELEVENLABS_API_KEY: print("Error: ELEVENLABS_API_KEY not found in environment variables.")
if not GOOGLE_API_KEY: print("Error: GOOGLE_API_KEY not found in environment variables.")
if not MAPS_API_KEY: print("Error: MAPS_API_KEY not found in environment variables.")
# --- End API Key Validation ---

VOICE_ID = 'pFZP5JQG7iQjIQuC4Bku'

FORMAT = pyaudio.paInt16
CHANNELS = 1
# SEND_SAMPLE_RATE = 16000 # Keep if used by RealtimeSTT or other input processing
RECEIVE_SAMPLE_RATE = 24000 # For ElevenLabs output
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

        # --- Initialize Google GenAI Client ---
        self.client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1beta'})
        self.model = "gemini-2.0-flash-live-001"

        # --- System Behavior Prompt (Updated from reference) ---
        self.system_behavior = """
            Your name is Ada, which stands for Advanced Design Assistant.
            You have a joking personality. You are an AI designed to assist with engineering projects, and you are an expert in all engineering, math, and science disciplines.
            You address people as "Sir" and you also speak with a british accent.
            When answering, you respond using complete sentences and in a conversational tone. Make sure to keep tempo of answers quick so don't use too much commas, periods or overall punctuation.
            Any prompts that need current or recent data always use the search tool.
            """
        
        # --- Function Declarations (Added get_travel_duration_func) ---
        self.get_weather_func = types.FunctionDeclaration(
            name="get_weather",
            description="Get the current weather conditions (temperature, precipitation, description) for a specified city and state/country (e.g., 'Vinings, GA', 'London, UK').",
            parameters=types.Schema(
                type=types.Type.OBJECT, properties={"location": types.Schema(type=types.Type.STRING, description="The city and state, e.g., San Francisco, CA or Vinings, GA")}, required=["location"]
            )
        )
        self.get_travel_duration_func = types.FunctionDeclaration(
            name="get_travel_duration",
            description="Calculates the estimated travel duration between a specified origin and destination using Google Maps. Considers current traffic for driving mode.",
            parameters=types.Schema(
                type=types.Type.OBJECT, properties={
                    "origin": types.Schema(type=types.Type.STRING, description="The starting address or place name."),
                    "destination": types.Schema(type=types.Type.STRING, description="The destination address or place name."),
                    "mode": types.Schema(type=types.Type.STRING, description="Optional: Mode of transport ('driving', 'walking', etc.). Defaults to 'driving'.")
                }, required=["origin", "destination"]
            )
        )
        # --- End Function Declarations ---

        # --- Map function names to actual methods (Added get_travel_duration) ---
        self.available_functions = {
            "get_weather": self.get_weather,
            "get_travel_duration": self.get_travel_duration # Added mapping
        }

        # --- Google Search Tool (Grounding) ---
        self.google_search_tool = Tool(
            google_search = GoogleSearch()
        )

        # --- Configuration (Updated tools list) ---
        self.config = types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=self.system_behavior)]
            ),
            response_modalities=["TEXT"],
            # ---> Updated tools list <---
            tools=[self.google_search_tool, types.Tool(code_execution=types.ToolCodeExecution,function_declarations=[
                self.get_weather_func,
                self.get_travel_duration_func # Add the new function here
                ])]
        )
        # --- End Configuration ---

        # --- Queues (Kept original relevant queues) ---
        self.input_queue = asyncio.Queue()
        self.response_queue = asyncio.Queue()
        self.audio_queue = asyncio.Queue() # Renamed from audio_output_queue for consistency

        # --- Recorder Config (Kept original) ---
        self.recorder_config = {
            'model': 'large-v3',
            'spinner': False,
            'language': 'en',
            'silero_sensitivity': 0.01,
            'webrtc_sensitivity': 3,
            'post_speech_silence_duration': 0.1,
            'min_length_of_recording': 0.2,
            'min_gap_between_recordings': 0,
        }

        # --- Initialize Recorder and PyAudio (Kept original) ---
        try:
            self.recorder = AudioToTextRecorder(**self.recorder_config)
        except Exception as e:
            print(f"Error initializing AudioToTextRecorder: {e}")
            self.recorder = None

        try:
            self.pya = pyaudio.PyAudio()
        except Exception as e:
            print(f"Error initializing PyAudio: {e}")
            self.pya = None
        # --- End Initialization ---

    # --- Function Implementations ---

    async def get_weather(self, location: str) -> dict | None:
        """ Fetches current weather. (Removed SocketIO emit) """
        async with python_weather.Client(unit=python_weather.IMPERIAL) as client:
            try:
                weather = await client.get(location)
                weather_data = {
                    'location': location,
                    'current_temp_f': weather.temperature,
                    'precipitation': weather.precipitation,
                    'description': weather.description,
                }
                print(f"Weather data fetched: {weather_data}")
                # --- SocketIO Emit Removed ---
                return weather_data # Return data for Gemini

            except Exception as e:
                print(f"Error fetching weather for {location}: {e}")
                return {"error": f"Could not fetch weather for {location}."} # Return error info

    # --- Added Travel Duration Functions (from reference, removed SocketIO emit) ---
    def _sync_get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> str:
        """ Synchronous helper for Google Maps API call """
        if not MAPS_API_KEY or MAPS_API_KEY == "YOUR_PROVIDED_KEY": # Check the actual key
            print("Error: Google Maps API Key is missing or invalid.")
            return "Error: Missing or invalid Google Maps API Key configuration."
        try:
            gmaps = googlemaps.Client(key=MAPS_API_KEY) # Use the loaded key
            now = datetime.now()
            print(f"Requesting directions: From='{origin}', To='{destination}', Mode='{mode}'")
            directions_result = gmaps.directions(origin, destination, mode=mode, departure_time=now)
            if directions_result:
                leg = directions_result[0]['legs'][0]
                duration_text = "Not available"
                result = f"Duration information not found in response for {mode}." # Default result
                if mode == "driving" and 'duration_in_traffic' in leg:
                    duration_text = leg['duration_in_traffic']['text']
                    result = f"Estimated travel duration ({mode}, with current traffic): {duration_text}"
                elif 'duration' in leg:
                    duration_text = leg['duration']['text']
                    result = f"Estimated travel duration ({mode}): {duration_text}"

                print(f"Directions Result: {result}")
                return result
            else:
                print(f"No route found from {origin} to {destination} via {mode}.")
                return f"Could not find a route from {origin} to {destination} via {mode}."
        except googlemaps.exceptions.ApiError as api_err:
             print(f"Google Maps API Error: {api_err}")
             return f"Error contacting Google Maps: {api_err}"
        except Exception as e:
            print(f"An unexpected error occurred during travel duration lookup: {e}")
            return f"An unexpected error occurred: {e}"

    async def get_travel_duration(self, origin: str, destination: str, mode: str = "driving") -> dict:
        """ Async wrapper to get travel duration. (Removed SocketIO emit) """
        print(f"Received request for travel duration from: {origin} to: {destination}, Mode: {mode}")
        if not mode:
            mode = "driving"

        try:
            result_string = await asyncio.to_thread(
                self._sync_get_travel_duration, origin, destination, mode
            )
            # --- SocketIO Emit Removed ---
            return {"duration_result": result_string} # Return result for Gemini

        except Exception as e:
            print(f"Error calling _sync_get_travel_duration via to_thread: {e}")
            return {"duration_result": f"Failed to execute travel duration request: {e}"}
    # --- End Travel Duration Functions ---


    async def clear_queues(self, text=""):
        """Clears all data from the input, response, and audio queues."""
        # Changed audio_queue name for consistency
        queues = [self.input_queue, self.response_queue, self.audio_queue]
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    break  # Queue is empty

    async def input_message(self):
        """ Handles user text input (Kept original) """
        while True:
            try:
                prompt = await asyncio.to_thread(input, "Enter your message: ")
                if prompt.lower() == "exit":
                    await self.input_queue.put("exit")  # Signal to exit
                    print("exit input")
                    break
                await self.clear_queues()
                await self.input_queue.put(prompt)
            except Exception as e:
                print(f"Error in input_message: {e}")
                continue  # Continue the loop even if there's an error

    # --- send_prompt: Updated with Function Calling/Grounding logic from reference ---
    async def send_prompt(self):
        """Manages the Gemini conversation session, handling text and tool calls."""
        print("Starting Gemini session manager...")
        try:
            # Establish connection (same as original)
            async with self.client.aio.live.connect(model=self.model, config=self.config) as session:
                print("Gemini session connected.")

                while True: # Loop to process text inputs
                    message = await self.input_queue.get()

                    if message.lower() == "exit":
                        print("Exit signal received in send_prompt.")
                        break # Exit the main loop

                    if not session: # Check session validity (though handled by async with)
                        print("Gemini session is not active.")
                        self.input_queue.task_done(); continue # Should not happen here

                    # Send the final text input for the turn (same as original)
                    print(f"Sending FINAL text input to Gemini: {message}")
                    await session.send(input=message, end_of_turn=True)
                    print("Final text message sent to Gemini, waiting for response...")

                    # --- Process responses (NEW LOGIC based on reference) ---
                    async for response in session.receive():
                        try:
                            # --- Handle Tool Calls (Function Calling) ---
                            if response.tool_call:
                                function_call_details = response.tool_call.function_calls[0]
                                tool_call_id = function_call_details.id
                                tool_call_name = function_call_details.name
                                tool_call_args = dict(function_call_details.args)

                                print(f"--- Received Tool Call: {tool_call_name} with args: {tool_call_args} (ID: {tool_call_id}) ---")

                                if tool_call_name in self.available_functions:
                                    function_to_call = self.available_functions[tool_call_name]
                                    try:
                                        # Execute the corresponding async function
                                        function_result = await function_to_call(**tool_call_args)

                                        # Construct the response to send back to Gemini
                                        func_resp = types.FunctionResponse(
                                            id=tool_call_id,
                                            name=tool_call_name,
                                            response={"content": function_result} # Send back the result dictionary
                                        )
                                        print(f"--- Sending Tool Response for {tool_call_name} (ID: {tool_call_id}) ---")
                                        # Send the function result back, don't end the turn yet
                                        await session.send(input=func_resp, end_of_turn=False)

                                    except Exception as e:
                                        print(f"Error executing function {tool_call_name}: {e}")
                                        # Decide how to handle function execution errors (e.g., send error back?)
                                        # For now, just print and continue waiting for Gemini's next step
                                else:
                                    print(f"Error: Unknown function called by Gemini: {tool_call_name}")
                                    # Decide how to handle unknown function calls
                                continue # Move to next response chunk after handling tool call

                            # --- Handle Text Responses ---
                            elif response.text:
                                text_chunk = response.text
                                print(text_chunk, end="", flush=True) # Print chunk immediately (like original)
                                await self.response_queue.put(text_chunk) # Put chunk onto queue for TTS

                            # --- (Optional) Handle Executable Code Tool (like reference, no SocketIO) ---
                            elif (response.server_content and
                                  response.server_content.model_turn and
                                  response.server_content.model_turn.parts and
                                  response.server_content.model_turn.parts[0].executable_code):
                                try:
                                    executable_code = response.server_content.model_turn.parts[0].executable_code
                                    code_string = executable_code.code
                                    language = str(executable_code.language) # Get language as string
                                    print(f"\n--- Received Executable Code ({language}) ---")
                                    print(code_string)
                                    print("------------------------------------------")
                                    # NOTE: No execution here, just printing. The library handles execution if configured.
                                except (AttributeError, IndexError, TypeError) as e:
                                    pass # Ignore errors if structure isn't as expected

                        except Exception as e:
                             print(f"\nError processing Gemini response chunk: {e}")
                             # Potentially break or continue depending on severity
                    # --- End Processing Responses ---

                    print("\nEnd of Gemini response stream for this turn.")
                    await self.response_queue.put(None) # Signal end of response for TTS
                    self.input_queue.task_done() # Mark input processed

        except asyncio.CancelledError:
            print("Gemini session task cancelled.")
        except Exception as e:
            print(f"Error in Gemini session manager: {e}")
        finally:
            print("Gemini session manager finished.")
            # No specific cleanup needed here unless tasks were managed differently

    async def tts(self):
        """ Send text to ElevenLabs API and stream the returned audio. (Kept Original Logic)"""
        uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream-input?model_id=eleven_flash_v2_5&output_format=pcm_24000"
        while True: # Outer loop to handle reconnections
            print("Attempting to connect to ElevenLabs WebSocket...")
            try:
                async with websockets.connect(uri) as websocket:
                    print("ElevenLabs WebSocket Connected.")
                    try:
                        # Send initial configuration
                        await websocket.send(json.dumps({
                            "text": " ",
                            "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "speed": 1.1},
                            "xi_api_key": ELEVENLABS_API_KEY,
                        }))

                        async def listen():
                            """Listen to the websocket for audio data and queue it."""
                            while True:
                                try:
                                    message = await websocket.recv()
                                    data = json.loads(message)
                                    if data.get("audio"):
                                        # Put raw audio bytes onto the queue
                                        await self.audio_queue.put(base64.b64decode(data["audio"]))
                                    elif data.get("isFinal"):
                                        # Optional: Handle end-of-stream signal from ElevenLabs if needed
                                        pass
                                    # Removed `elif text is None:` check as it was incorrect scope
                                except websockets.exceptions.ConnectionClosedOK:
                                    print("ElevenLabs connection closed normally by server.")
                                    break # Exit listener loop
                                except websockets.exceptions.ConnectionClosedError as e:
                                     print(f"ElevenLabs connection closed with error: {e}")
                                     break # Exit listener loop
                                except json.JSONDecodeError as e:
                                    print(f"JSON Decode Error in ElevenLabs listener: {e}")
                                    # Decide whether to break or continue
                                except asyncio.CancelledError:
                                     print("ElevenLabs listener task cancelled.")
                                     raise # Re-raise cancellation
                                except Exception as e:
                                    print(f"Error in ElevenLabs listener: {e}")
                                    break # Exit listener loop on other errors

                        listen_task = asyncio.create_task(listen())

                        try:
                            # Send text chunks from response queue
                            while True:
                                text = await self.response_queue.get()
                                if text is None: # Signal to end the TTS stream for this turn
                                    print("End of text stream signal received for TTS.")
                                    await websocket.send(json.dumps({"text": ""})) # Send EOS signal
                                    break # Exit inner loop (sending text)

                                if text: # Ensure text is not empty
                                    # Added space for potential word breaks
                                    await websocket.send(json.dumps({"text": text + " "}))

                                self.response_queue.task_done() # Mark item as processed

                        except asyncio.CancelledError:
                            print("TTS text sender cancelled.")
                            listen_task.cancel() # Cancel listener if sender is cancelled
                            raise # Re-raise cancellation
                        except Exception as e:
                            print(f"Error processing text for TTS: {e}")
                            listen_task.cancel() # Cancel listener on error
                        finally:
                            # Wait for the listener task to finish after text sending stops or errors
                            if not listen_task.done():
                                print("Waiting for TTS listener task to complete...")
                                try:
                                    await asyncio.wait_for(listen_task, timeout=5.0)
                                except asyncio.TimeoutError:
                                    print("Timeout waiting for TTS listener task.")
                                    listen_task.cancel()
                                except asyncio.CancelledError:
                                     print("TTS Listener was already cancelled.") # Expected if sender was cancelled
                                except Exception as e:
                                     print(f"Error awaiting listener task: {e}")


                    except websockets.exceptions.ConnectionClosed as e:
                         print(f"ElevenLabs WebSocket connection closed during operation: {e}")
                         # Outer loop will handle reconnection attempt
                    except Exception as e:
                        print(f"Error during ElevenLabs websocket communication: {e}")
                        # Outer loop will handle reconnection attempt

            except websockets.exceptions.WebSocketException as e:
                print(f"ElevenLabs WebSocket connection failed: {e}")
            except asyncio.CancelledError:
                 print("TTS main task cancelled.")
                 break # Exit outer loop if cancelled
            except Exception as e:
                print(f"Error connecting to ElevenLabs websocket: {e}")

            print("Waiting 5 seconds before attempting ElevenLabs reconnection...")
            await asyncio.sleep(5) # Wait before retrying connection

    # Removed extract_tool_call method as it's replaced by direct handling in send_prompt

    async def play_audio(self):
        """ Plays audio chunks from the audio_queue. (Kept Original Logic) """
        if self.pya is None:
            print("PyAudio is not initialized. Cannot play audio.")
            return

        stream = None # Initialize stream variable
        try:
            print("Opening PyAudio stream...")
            stream = await asyncio.to_thread(
                self.pya.open,
                format=FORMAT,
                channels=CHANNELS,
                rate=RECEIVE_SAMPLE_RATE,
                output=True,
            )
            print("PyAudio stream opened. Waiting for audio chunks...")
            while True:
                try:
                    # Wait for audio data from the TTS task
                    bytestream = await self.audio_queue.get()
                    if bytestream is None: # Potential signal to stop? (Not currently used)
                         print("Received None in audio queue, stopping playback loop.")
                         break
                    # Write audio data to the stream in a separate thread
                    await asyncio.to_thread(stream.write, bytestream)
                    self.audio_queue.task_done() # Mark item as processed
                except asyncio.CancelledError:
                    print("Audio playback task cancelled.")
                    break  # Exit loop if task is cancelled
                except Exception as e:
                    print(f"Error in play_audio loop: {e}")
                    # Decide if error is fatal or recoverable
                    await asyncio.sleep(0.1) # Avoid busy-looping on error

        except pyaudio.PyAudioError as e:
            print(f"PyAudio error opening stream: {e}")
        except Exception as e:
            print(f"Error setting up audio stream: {e}")
        finally:
            if stream:
                print("Closing PyAudio stream...")
                await asyncio.to_thread(stream.stop_stream)
                await asyncio.to_thread(stream.close)
                print("PyAudio stream closed.")
            # Don't terminate PyAudio here if other parts might use it
            # await asyncio.to_thread(self.pya.terminate)

    async def stt(self):
        """ Listens via microphone and puts transcribed text onto input_queue. (Kept Original Logic) """
        if self.recorder is None:
            print("Audio recorder (RealtimeSTT) is not initialized.")
            return

        print("Starting Speech-to-Text engine...")
        while True:
            try:
                # Blocking call handled in a thread
                text = await asyncio.to_thread(self.recorder.text)
                if text: # Only process if text is not empty
                    print(f"STT Detected: {text}")
                    await self.clear_queues() # Clear queues before adding new input
                    await self.input_queue.put(text) # Put transcribed text onto the input queue
            except asyncio.CancelledError:
                 print("STT task cancelled.")
                 break
            except Exception as e:
                print(f"Error in STT loop: {e}")
                # Add a small delay to prevent high CPU usage on continuous errors
                await asyncio.sleep(0.5)
# --- End of ADA Class ---

# --- Main Execution Block (Example) ---
async def main():
    print("Starting Ada Assistant...")
    ada = ADA()

    if ada.pya is None or ada.recorder is None:
         print("Failed to initialize audio components. Exiting.")
         return

    # Create tasks for each concurrent operation
    tasks = [
        asyncio.create_task(ada.stt()),          # Speech to Text -> input_queue
        asyncio.create_task(ada.send_prompt()),  # input_queue -> Gemini (handles tools) -> response_queue
        asyncio.create_task(ada.tts()),          # response_queue -> ElevenLabs -> audio_queue
        asyncio.create_task(ada.play_audio()),   # audio_queue -> Speaker
        # asyncio.create_task(ada.input_message()) # Optional: Uncomment for text input instead of STT
    ]

    # Run tasks concurrently
    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        print("Main tasks cancelled.")
    finally:
         print("Cleaning up...")
         # Gracefully stop tasks if needed (though gather handles cancellation)
         for task in tasks:
              if not task.done():
                   task.cancel()
         await asyncio.gather(*tasks, return_exceptions=True) # Wait for cleanup
         if ada.pya:
              print("Terminating PyAudio.")
              await asyncio.to_thread(ada.pya.terminate) # Clean up PyAudio resources

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting Ada Assistant...")
    except Exception as e:
         print(f"\nAn unexpected error occurred in main: {e}")
