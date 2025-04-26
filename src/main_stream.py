from nicegui import ui
from datetime import datetime
import asyncio
import src.inference as inference
import time 
import re
import argparse
import logging
from logging.handlers import RotatingFileHandler
import json
import os

logger = logging.getLogger(__name__)

class _AsyncGenWrapper:
    """
    Wraps a normal (sync) generator so that you can do:
        async for item in _AsyncGenWrapper(sync_gen): ...
    """
    def __init__(self, sync_gen):
        self._gen = sync_gen

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            # Run next() in a thread so we don’t block the event loop
            return await asyncio.to_thread(self._gen.__next__)
        except StopIteration:
            raise StopAsyncIteration

# -------- Module-level logger----------------
def configure_logging(debug: bool = False):    
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            RotatingFileHandler("app.log", 
                                maxBytes=1e7, 
                                backupCount=3,
                                mode="w"),
        ],
        force=True
    )
    logging.getLogger("llama_cpp").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

# --------- ChatWindows class ---------------
class ChatWindows:
    def __init__(self, bot: inference.PleiasBot):
        self.bot = bot
        
        # Populated by UI
        self.input_field = None
        self.chat_display = None
        self.analysis_container = None
        
        # For future
        self.user_avatar = None
        self.bot_avatar = None
        
    async def send_message(self):
        message_input = self.input_field.value.strip()
        if not message_input:
            return
    
        with self.chat_display:
            ui.chat_message(text=message_input, 
                            stamp=datetime.now().strftime('%X'), 
                            avatar=self.user_avatar, sent=True
                            ).props('bg-color=secondary')
            
        self.input_field.value = ""  # Clear input field
        
        start = time.time()
        with self.chat_display:
            placeholder = ui.chat_message(
                text="Thinking...",
                # stamp=datetime.now().strftime('%X'),
                avatar=self.bot_avatar,
                sent=False
            ).props('bg-color=grey-2 text-black')
            
            
            bot_message = ui.chat_message(
                text="...",
                # stamp=datetime.now().strftime('%X'),
                avatar=self.bot_avatar,
                sent=False
            ).props('bg-color=grey-2 text-black')
        bot_message.visible = False 
        
        # prepare the area where we’ll stream the answer

        raw_buffer = ""
        seen_answer_start = False

        # now consume the raw stream
        async for packet in _AsyncGenWrapper(self.bot.stream_predict(message_input)):
            # when we get the final dict, packet is a dict with "__done__"
            if isinstance(packet, dict) and "__done__" in packet:
                final = packet["__done__"]
                sources = packet["sources"]
                # swap in the fully formatted answer
                bot_message.clear()
                with bot_message:
                    ui.html(self.render_with_tooltips(final["answer"]))

                # render the sources / analysis
                self.analysis_container.clear()
                with self.analysis_container:
                    ui.markdown("**Sources**")
                    self.display_sources(sources)
                    ui.separator()
                    ui.html(f"<details><summary>Analysis</summary>{final.get('source_analysis','')}</details>")
                break

            # otherwise packet is just a string chunk
            raw_buffer += packet

            if not seen_answer_start:
                # look for the delimiter
                idx = raw_buffer.find("<|answer_start|>")
                if idx != -1:
                    seen_answer_start = True
                    # everything _after_ the marker is the first bit of the answer
                    visible = raw_buffer[idx + len("<|answer_start|>"):]
                    bot_message.clear()
                    bot_message.visible = True
                    with bot_message:
                        ui.html(self.render_with_tooltips(visible))
                    
                    placeholder.clear()
                    elapsed_time = time.time() - start
                    with placeholder:
                        ui.label(f"Thought for {elapsed_time} seconds")
                # else: still in reasoning phase, do nothing

            else:
                # piecemeal update
                bot_message.clear()
                with bot_message:
                    ui.html(self.render_with_tooltips(
                    raw_buffer.split("<|answer_start|>",1)[1]
                ))
                

            await asyncio.sleep(0.01)   # allow the front-end to repaint
            
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')


    async def get_response(self, message_input: str) -> str:
        if message_input:
            results = await asyncio.to_thread(self.bot.predict, message_input)
            return results["processed"]["source_analysis"], results["processed"]["answer"], results["sources"]
        else:
            return ""
        
    
    def render_with_tooltips(self, text: str) -> str:
        """
        Replace every <ref name="<|source_id|>N">tooltip text</ref> in `text`
        with an HTML <span> that shows [N] and on hover displays the tooltip text.
        """
        # This pattern finds each <ref …>…</ref>
        pattern = re.compile(
            r'<ref\s+name="<\|source_id\|>(\d+)">(.*?)</ref>',
            re.DOTALL
        )

        parts = []
        last_end = 0

        for m in pattern.finditer(text):
            # append the text up to this ref
            parts.append(text[last_end : m.start()])
            source_id   = m.group(1)
            tooltip_txt = m.group(2).replace('"', "&quot;")  # escape quotes

            # build the HTML span
            parts.append(
                f' <span class="tooltip fade" data-title="{tooltip_txt}" '
                'style="color:black;text-decoration:underline;">'
                f'[{source_id}]</span>'
            )
            last_end = m.end()

        # append any trailing text after the final ref
        parts.append(text[last_end:])
        full_text = "".join(parts)
        return f'<div style="white-space: pre-wrap;">{full_text}</div>'

    
    def display_sources(self, sources):
        for source in sources:
            ui.html(f"""<a href={source["metadata"]["url"]} target="_blank" style="color: blue; text-decoration: underline;">
                    [Source {source['id']}] <br>
                    </a>
                    {source["metadata"]["section"]} {source["metadata"]["subsection1"]} {source["metadata"]["subsection2"]} {source["metadata"]["subsection3"]}
                    <br> {source['text']}
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
        chat_display.visible = (tab_name == "Main")
        library.visible = (tab_name == "Library")
        conv.visible = (tab_name == "Conversations")
    

    # HEADER with Tabs
    with ui.header(elevated=True).classes('justify-between w-full p-4 bg-[#B63838]') as header:
        chat_windows.header = header
        with ui.row():
            ui.image('assets/pleias-no-background.png').classes('h-8 w-8')
            ui.button('Main', on_click=lambda: switch_tab("Main")).props('shadow-none')
            ui.button('Library', on_click=lambda: switch_tab("Library")).classes('shadow-none')
            ui.button('Conversations', on_click=lambda: switch_tab("Conversations"))
        ui.space()
        ui.button('About', on_click=lambda: ui.notify("About Page"))
        ui.button('Log In', on_click=lambda: ui.notify("Log In"))
        with ui.button(on_click=lambda: analysis_drawer.toggle(), icon='menu') as drawer_button: # color="#5A3D97"
            ui.tooltip('See Sources')        


    # LIBRARY & CONVERSATIONS (initially hidden)
    with ui.column().classes('w-full hidden') as library:
        ui.label('Library Section - Work in progress...')

    with ui.column().classes('w-full hidden') as conv:
        ui.label('Conversations Section - Work in progress...')

    # CHAT ARENA
    with ui.column().classes('w-full max-w-2xl mx-auto items-stretch') as chat_display:
        pass
    chat_windows.chat_display = chat_display
    
    # INPUT BAR
    with ui.footer().classes('bg-white') as query_bar:
        with ui.row().classes('w-full max-w-3xl mx-auto my-2 no-wrap items-center'):
            with ui.avatar():
                ui.image(chat_windows.user_avatar).classes(f'rounded-full bg-[{main_colour}]')
            chat_windows.input_field = ui.input(placeholder="Type your question...") \
                .props('rounded outlined input-class=mx-3').classes('flex-grow') \
                .on('keydown.enter', chat_windows.send_message)
            ui.button("Send", on_click=chat_windows.send_message)
            
    # ANAL
    with ui.right_drawer(fixed=False, value=False).style(f'background-color: {pale_colour}') as analysis_drawer:
        with ui.column() as analysis_container:
            chat_windows.analysis_container = analysis_container
            ui.label('Ask a question to see the sources and analysis')
            
    analysis_drawer.bind_visibility_from(chat_display)
    query_bar.bind_visibility_from(chat_display)

# --------- Main function ---------------

def main():
    # Rewrite log file if it exists
    if os.path.exists("app.log"):
        open("app.log", "w").close()
            
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("table_name", nargs="?", default="both_fts",
                        help="Table name (default: both_fts)")
    
    parser.add_argument(
        "-t", "--table-name",
        dest="t",
        default="both_fts",
        help="Table name (default: %(default)s)"
    )
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging and bot
    configure_logging(args.debug)
    bot = inference.PleiasBot(args.table_name)
    
    # Intantiate ChatWindows and UI
    chat_windows = ChatWindows(bot)
    setup_ui(chat_windows)
    
    # 4. run NiceGUI
    ui.run(host="0.0.0.0", port=8081)

if __name__ in {"__main__", "__mp_main__"}:
    main()