import argparse
import asyncio
import logging
import os
import re
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List

from nicegui import ui

import src.inference as inference

logger = logging.getLogger(__name__)


# -------- Module-level logger----------------
def configure_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler("app.log", maxBytes=1e7, backupCount=3, mode="w"),
        ],
        force=True,
    )
    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


# --------- AsyncGenWrapper helper---------------
class _AsyncGenWrapper:
    """
    Helper that wraps a normal (sync) generator to make it async iterable.
    """

    def __init__(self, sync_gen):
        self._gen = sync_gen

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Offload the blocking next() call to a thread
            return await asyncio.to_thread(self._gen.__next__)
        except StopIteration:
            # Signal the end of the async iteration
            raise StopAsyncIteration


# --------- ChatWindows class ---------------
class ChatWindows:
    """
    A class to handle chat interface interactions between a user and a Pleias bot.
    This class manages the UI components for chat, including message sending,
    response rendering, and source display. It supports both streaming and static
    response modes.
    Attributes:
        bot (inference.PleiasBot): The bot instance handling inference.
        stream (bool): Whether to use streaming responses (default: False).
        input_field: UI text input component for user messages (populated by UI).
        chat_display: UI container for displaying chat messages (populated by UI).
        analysis_container: UI container for displaying analysis and sources (populated by UI).
        user_avatar: User avatar graphic for the chat interface.
        bot_avatar: Bot avatar graphic for the chat interface.
        _tooltip_pattern (re.Pattern): Regular expression for parsing source references.
    Methods:
        send_message: Process user input and display bot response.
        _send_message_static: Handle non-streaming response mode.
        _send_message_streaming: Handle streaming response mode.
        _safe_streamed_text: Clean up partially streamed text with incomplete references.
        render_with_tooltips: Convert source references to HTML tooltips.
        display_sources: Render source information in the UI.
    """

    def __init__(self, bot: inference.PleiasBot, stream: bool = False):
        self.bot = bot
        self.stream = stream
        self._tooltip_pattern = re.compile(
            r'<ref\s+name="<\|source_id\|>(\d+)">(.*?)</ref>', re.DOTALL
        )

        # Populated by UI
        self.input_field = None
        self.chat_display = None
        self.analysis_container = None

        # For future
        self.user_avatar = None
        self.bot_avatar = None

    async def send_message(self) -> None:
        """
        Process and send a user message to the chat interface.
        This method handles the following steps:
        1. Get the message input from the input field
        2. Display the user's message in the chat interface
        3. Clear the input field
        4. Show a "Thinking..." placeholder while waiting for the bot's response
        5. Create a hidden bot response message that will be updated
        6. Get and display the bot's response (either streaming or static)
        7. Auto-scroll to the latest message
        The method handles both streaming and non-streaming response modes based on
        the self.stream attribute.
        Returns:
            None
        """

        message_input = self.input_field.value.strip()
        if not message_input:
            return

        # Append user message inside the vertical column
        with self.chat_display:
            ui.chat_message(
                text=message_input,
                stamp=datetime.now().strftime("%X"),
                avatar=self.user_avatar,
                sent=True,
            ).props("bg-color=secondary")

        self.input_field.value = ""  # Clear input field

        with self.chat_display:
            # The placeholder message with reasoning
            thinking_message = ui.chat_message(
                text="Thinking...",
                # stamp=datetime.now().strftime('%X'),
                avatar=self.bot_avatar,
                sent=False,
            ).props("bg-color=grey-2 text-black")

            # The message that will be updated with the answer
            bot_response = (
                ui.chat_message(
                    text="...",
                    # stamp=datetime.now().strftime('%X'),
                    avatar=self.bot_avatar,
                    sent=False,
                    text_html=True,
                )
                .props("bg-color=grey-2 text-black")
                .style("white-space: pre-wrap; line-height:1.5;")
            )
            bot_response.visible = False

        # Get bot response
        logger.info("Getting response")
        if self.stream:
            await self._send_message_streaming(
                message_input, bot_response, thinking_message
            )
        else:
            await self._send_message_static(
                message_input, bot_response, thinking_message
            )

        # Auto-scroll to latest message
        ui.run_javascript("window.scrollTo(0, document.body.scrollHeight)")

    async def _send_message_static(
        self,
        message_input: str,
        bot_response: ui.chat_message,
        thinking_message: ui.chat_message,
    ) -> None:
        """
        Sends a message to the bot and displays the response in a static workflow.
        This method handles sending the user's input to the bot, getting the response,
        and updating the UI with the response content, elapsed time, sources, and analysis.
        Args:
            message_input (str): The user's message to send to the bot.
            bot_response (ui.chat_message): The UI element to display the bot's response.
            thinking_message (ui.chat_message): The UI element to display thinking time information.
        Returns:
            None: This method updates the UI components but does not return a value.
        Note:
            This method processes the request synchronously and updates various UI components:
            - The bot's response text with rendered tooltips
            - The thinking time indicator
            - The sources of information used
            - A collapsible analysis section
        """

        start = time.time()

        # Get the bot's response (blocking call)
        results = await asyncio.to_thread(self.bot.predict, message_input)
        analysis_text, answer_text, sources = (
            results["processed"]["source_analysis"],
            results["processed"]["answer"],
            results["sources"],
        )

        elapsed_time = int(time.time() - start)
        with self.chat_display:
            # Update response
            bot_response.clear()
            bot_response.visible = True
            with bot_response:
                ui.html(self.render_with_tooltips(answer_text))

            # Update thinking message with elapsed time
            thinking_message.clear()
            elapsed_time = int(time.time() - start)
            with thinking_message:
                ui.label(f"Thought for {elapsed_time} seconds")

        # Render sources and analysis
        self.analysis_container.clear()
        with self.analysis_container:
            ui.markdown("**Sources**")
            self.display_sources(sources)
            ui.separator()
            ui.html(f"""
                <details>
                    <summary><strong>Analysis</strong></summary>
                    {analysis_text}
                </details>""")

    async def _send_message_streaming(
        self,
        message_input: str,
        bot_response: ui.chat_message,
        thinking_message: ui.chat_message,
    ) -> None:
        """
        Asynchronously streams the bot's response to the user's message with visual feedback.
        This method processes the streaming output from the bot, handling different phases:
        1. Reasoning phase before the answer starts
        2. Answer generation phase
        3. Final completion with sources and analysis
        The method updates the UI in real-time to show thinking progress, streamed answer text,
        and eventually the final formatted response with sources.
        Args:
            message_input (str): The user's input message to process
            bot_response (ui.chat_message): UI element to display the bot's response
            thinking_message (ui.chat_message): UI element to display thinking status
        Returns:
            None: The method updates UI components directly instead of returning values
        Note:
            This implementation relies on the bot's stream_predict method returning
            either string chunks during generation or a dictionary with "__done__" key
            containing the final response and sources when complete.
        """

        start = time.time()
        raw_buffer = ""  # Track the whole generated text
        answer_buffer = ""  # Track the answer text (after <|answer_start|>)
        last_clean_answer = (
            ""  # Track the last clean answer (to update UI only when needed)
        )
        seen_answer_start = False

        # Consume the generator in a loop
        async for packet in _AsyncGenWrapper(self.bot.stream_predict(message_input)):
            # Case 1: we are in the final phase, we get a dict with "__done__"
            if isinstance(packet, dict) and "__done__" in packet:
                final = packet["__done__"]
                sources = packet["sources"]
                # swap in the fully formatted answer
                bot_response.clear()
                with bot_response:
                    ui.html(self.render_with_tooltips(final["answer"]))

                # Render the sources / analysis
                self.analysis_container.clear()
                with self.analysis_container:
                    ui.markdown("**Sources**")
                    self.display_sources(sources)
                    ui.separator()
                    ui.html(
                        f"<details><summary>Analysis</summary>{final.get('source_analysis', '')}</details>"
                    )
                break

            # Otherwise, we generation is in progress and we get a string
            raw_buffer += packet

            # Case 2: we are still in the reasoning phase, so we look for the marker
            if not seen_answer_start:
                if "<|answer_start|>" in raw_buffer:
                    seen_answer_start = True
                    # slice out the answer part into its own buffer
                    answer_buffer = raw_buffer.split("<|answer_start|>", 1)[1]
                    bot_response.clear()
                    bot_response.visible = True
                    with bot_response:
                        ui.html(self.render_with_tooltips(answer_buffer))

                    thinking_message.clear()
                    elapsed_time = int(time.time() - start)
                    with thinking_message:
                        ui.label(f"Thought for {elapsed_time} seconds")

                    await asyncio.sleep(0)  # allow the front-end to repaint
                # Else: still in reasoning phase, do nothing

            # Case 3: we are in the answer phase, we update the message
            else:
                # We do not show while the reference is being built
                answer_buffer += packet
                clean_answer_buffer = self._safe_streamed_text(answer_buffer)

                if clean_answer_buffer != last_clean_answer:
                    last_clean_answer = clean_answer_buffer
                    bot_response.clear()
                    with bot_response:
                        ui.html(self.render_with_tooltips(clean_answer_buffer))
                    await asyncio.sleep(0)  # allow the front-end to repaint

    def _safe_streamed_text(self, raw: str) -> str:
        """
        Given the accumulated answer (everything after <|answer_start|>),
        return only the prefix up to the last *complete* <ref…>…</ref> block
        (or the whole string if there is no open-but-not-closed <ref>).
        """
        # find the last opening tag
        last_open = raw.rfind("<ref")
        # find the last closing tag
        last_close = raw.rfind("</ref>")
        # if there is an unclosed <ref…> (open after close), drop it
        if last_open > last_close:
            return raw[:last_open] + "<i> [Generating ref...]</i>"
        else:
            return raw

    def render_with_tooltips(self, text: str) -> str:
        """
        Converts reference tags in text to HTML tooltips with source indicators.
        This method searches for patterns like <ref name="<|source_id|>N">tooltip text</ref>
        in the input text and replaces them with HTML spans that display [N] in the text
        and show the tooltip content when hovered over.
        Args:
            text: Input string containing reference tags to be converted
        Returns:
            String with reference tags replaced by HTML tooltip spans
        Example:
            Input: "This is a <ref name=\"<|1|>1\">Source information</ref> citation."
            Output: "This is a <span class=\"tooltip fade\" data-title=\"Source information\"
                     style=\"color:black;text-decoration:underline;\">[1]</span> citation."
        """

        parts = []
        last_end = 0

        for m in self._tooltip_pattern.finditer(text):
            # append the text up to this ref
            parts.append(text[last_end : m.start()])
            source_id = m.group(1)
            tooltip_txt = m.group(2).replace('"', "&quot;")  # escape quotes

            # build the HTML span
            parts.append(
                f' <span class="tooltip fade" data-title="{tooltip_txt}" '
                'style="color:black;text-decoration:underline;">'
                f"[{source_id}]</span>"
            )
            last_end = m.end()

        # append any trailing text after the final ref
        parts.append(text[last_end:])
        full_text = "".join(parts)
        if not self.stream:
            logger.debug(f"Full HTML text:\n {full_text}")
        return full_text.strip()

    def display_sources(self, sources: List[Dict[str, Any]]) -> None:
        """
        Display a list of sources as clickable links with their associated metadata and text content.
        Each source is rendered as an HTML element containing:
        - A clickable link to the source URL with the source ID
        - The section hierarchy (section, subsection1, subsection2, subsection3)
        - The actual text content from the source
        Parameters:
        ----------
        sources : List[Dict[str, Any]]
            A list of source dictionaries, each containing:
            - 'id': unique identifier for the source
            - 'metadata': dictionary with keys:
                - 'url': link to the original source
                - 'section': main section title
                - 'subsection1': first level subsection title
                - 'subsection2': second level subsection title
                - 'subsection3': third level subsection title
            - 'text': the content text from the source
        Returns:
        -------
        None
            The function displays HTML content using ui.html() but doesn't return a value.
        """

        for source in sources:
            ui.html(f"""<a href={source["metadata"]["url"]} target="_blank" style="color: blue; text-decoration: underline;">
                    [Source {source["id"]}] <br>
                    </a>
                    {source["metadata"]["section"]} {source["metadata"]["subsection1"]} {source["metadata"]["subsection2"]} {source["metadata"]["subsection3"]}
                    <br> {source["text"]}
                    """)


# --------- Set up UI ---------------


def setup_ui(chat_windows: ChatWindows):
    # Basic setup
    main_colour = "#AB1717"
    pale_colour = "#FEECEC"
    button_hover_colour = "#D81D1D"

    ui.colors(primary=main_colour, secondary=pale_colour, accent=button_hover_colour)

    # HTML for tooltips
    ui.add_head_html("""
    <style>
    /* setup tooltips */
    .tooltip {
    position: relative;
    }
    .tooltip:before,
    .tooltip:after {
    display: block;
    opacity: 0;
    pointer-events: none;
    position: absolute;
    }

    .tooltip:before {
    background: rgba(0,0,0,.75);
    border-radius: 2px;
    color: #fff;
    content: attr(data-title);
    font-size: 12px; /* Reduced font size */
    padding: 4px 8px;
    top: 30px; /* Adjust to match arrow */
    left: 50%;
    transform: translateX(-50%); /* Center the tooltip */
    white-space: normal;
    width: 350px;
    }

    .tooltip:after {
    border-right: 6px solid transparent;
    border-bottom: 6px solid rgba(0,0,0,.75); 
    border-left: 6px solid transparent;
    content: '';
    height: 0;
    width: 0;
    top: 24px; /* Adjust to sit right below the tooltip box */
    left: 50%;
    transform: translateX(-50%); /* Center the arrow */
    }


    /* the animations */
    /* fade */
    .tooltip.fade:after,
    .tooltip.fade:before {
    transform: translate3d(0,-10px,0);
    transition: all .15s ease-in-out;
    }
    .tooltip.fade:hover:after,
    .tooltip.fade:hover:before {
    opacity: 1;
    transform: translate3d(0,0,0);
    }
    </style>
    """)

    # Helper to change tabs
    def switch_tab(tab_name):
        chat_display.visible = tab_name == "Main"
        library.visible = tab_name == "Library"
        conv.visible = tab_name == "Conversations"

    # HEADER with Tabs
    with ui.header(elevated=True).classes(
        "justify-between w-full p-4 bg-[#B63838]"
    ) as header:
        chat_windows.header = header
        with ui.row():
            ui.image("assets/pleias-no-background.png").classes("h-8 w-8")
            ui.button("Main", on_click=lambda: switch_tab("Main")).props("shadow-none")
            ui.button("Library", on_click=lambda: switch_tab("Library")).classes(
                "shadow-none"
            )
            ui.button("Conversations", on_click=lambda: switch_tab("Conversations"))
        ui.space()
        ui.button("About", on_click=lambda: ui.notify("About Page"))
        ui.button("Log In", on_click=lambda: ui.notify("Log In"))
        with ui.button(
            on_click=lambda: analysis_drawer.toggle(), icon="menu"
        ) as drawer_button:  # color="#5A3D97"
            ui.tooltip("See Sources")

    # LIBRARY & CONVERSATIONS (initially hidden)
    with ui.column().classes("w-full hidden") as library:
        ui.label("Library Section - Work in progress...")

    with ui.column().classes("w-full hidden") as conv:
        ui.label("Conversations Section - Work in progress...")

    # CHAT ARENA
    with ui.column().classes(
        "w-full max-w-2xl mx-auto items-stretch gap-0"
    ) as chat_display:
        pass
    chat_windows.chat_display = chat_display

    # INPUT BAR
    with ui.footer().classes("bg-white") as query_bar:
        with ui.row().classes("w-full max-w-3xl mx-auto my-2 no-wrap items-center"):
            with ui.avatar():
                ui.image(chat_windows.user_avatar).classes(
                    f"rounded-full bg-[{main_colour}]"
                )
            chat_windows.input_field = (
                ui.input(placeholder="Type your question...")
                .props("rounded outlined input-class=mx-3")
                .classes("flex-grow")
                .on("keydown.enter", chat_windows.send_message)
            )
            ui.button("Send", on_click=chat_windows.send_message)

    # ANALYSIS DRAWER
    with ui.right_drawer(fixed=False, value=False).style(
        f"background-color: {pale_colour}"
    ) as analysis_drawer:
        with ui.column() as analysis_container:
            chat_windows.analysis_container = analysis_container
            ui.label("Ask a question to see the sources and analysis")

    analysis_drawer.bind_visibility_from(chat_display)
    query_bar.bind_visibility_from(chat_display)


# --------- Main function ---------------


def main():
    """
    Main entry point for the Pleias application.
    Sets up command-line argument parsing for configuring the application,
    initializes logging, creates the bot for inference, and launches the
    chat interface UI.
    Command-line options:
        -t, --table-name: Database table name to use
        --debug: Enable debug logging
        --stream: Enable token-by-token streaming in UI
        --host: Host interface to bind the server to
        -p, --port: Port to listen on
    The application will clear any existing log file on startup.
    """

    # Rewrite log file if it exists
    if os.path.exists("app.log"):
        open("app.log", "w").close()

    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t",
        "--table-name",
        dest="table_name",
        default="both",
        help="Table name (default: %(default)s)",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable token-by-token streaming in the UI",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for the server (default: %(default)s)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        dest="port",
        default=8081,
        help="Port to listen on (default: %(default)s)",
    )

    args = parser.parse_args()

    # Setup logging and bot
    configure_logging(args.debug)
    bot = inference.PleiasBot(args.table_name)

    # Intantiate ChatWindows and UI
    chat_windows = ChatWindows(bot, stream=args.stream)
    setup_ui(chat_windows)

    # Run app
    ui.run(host=args.host, port=args.port)


if __name__ in {"__main__", "__mp_main__"}:
    main()
