from nicegui import ui
from datetime import datetime
import asyncio
import src.inference as inference
import time 
import re
import argparse
import logging


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("table_name", nargs="?", default="both_fts", help="Table name (default: both_fts)")
args = parser.parse_args()

pleias_bot = inference.pleiasBot(args.table_name)

main_colour = "#AB1717"
pale_colour = "#FEECEC"
button_hover_colour = "#D81D1D"

ui.colors(primary=main_colour, secondary=pale_colour, accent=button_hover_colour)


def switch_tab(tab_name):
    chat_display.visible = (tab_name == "Main")
    library_content.visible = (tab_name == "Library")
    conversations_content.visible = (tab_name == "Conversations")


class ChatWindows:
    def __init__(self):
        self.dialog_is_empty = True
        self.dialogue = []
        self.user_avatar = None
        self.bot_avatar = None

    async def send_message(self):
        message_input = input_field.value.strip()
        if not message_input:
            return

        # Append user message inside the vertical column
        with chat_display:
            ui.chat_message(text=message_input, 
                            stamp=datetime.now().strftime('%X'), 
                            avatar=self.user_avatar, sent=True
                            ).props('bg-color=secondary')
            
        input_field.value = ""  # Clear input field
        
        start = time.time()
        with chat_display:
            bot_message = ui.chat_message(
                text="Thinking...",
                # stamp=datetime.now().strftime('%X'),
                avatar=self.bot_avatar,
                sent=False
            ).props('bg-color=grey-2 text-black')
            
        # Get bot response
        logger.info("Getting response")
        analysis_text, answer_text, fiches_html, search_results = await self.get_response(message_input)  
        
        # Auto-scroll to latest message
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')
        
        elapsed_time = int(time.time() - start)
        bot_message.clear()
        with chat_display:
            # ui.chat_message(text=f"Thought for {elapsed_time} seconds").props('bg-color=grey-2 text-black')
            
            with ui.chat_message(
                stamp=datetime.now().strftime('%X'),
                avatar=self.bot_avatar,
                sent=False
            ).props('bg-color=grey-2 text-black') as bot_message:
                
                ui.label(f"Thought for {elapsed_time} seconds")
                logger.info("Search results:")
                logger.info(search_results.to_dict())
                with ui.label(): # main message
                    seen_hashes = self.display_format_references(answer_text, search_results)
                
                

        analysis_container.clear()
        with analysis_container:
            ui.markdown('**Sources**')
            self.display_sources(seen_hashes)
            ui.separator()
            ui.html(f"""
                <details>
                    <summary><strong>Analysis</strong></summary>
                    {analysis_text}
                </details>""")

        # Optionally, open the drawer automatically when new analysis is available
        # analysis_drawer.show()

        # Auto-scroll to latest message
        ui.run_javascript('window.scrollTo(0, document.body.scrollHeight)')


    async def get_response(self, message_input: str) -> str:
        if message_input:
            generated_text, fiches_html, search_results  = await asyncio.to_thread(pleias_bot.predict, message_input)
            analysis_text, answer_text = self.prepare_for_format(generated_text)
            return analysis_text, answer_text, fiches_html, search_results
        else:
            return ""
        
    def prepare_for_format(self, generated_text):
        # Split the text into analysis and answer sections
            parts = generated_text.split("<|source_analysis_end|>")
            if len(parts) == 2:
                analysis_text = parts[0].strip()
                answer_text = parts[1].replace("<|answer_start|>", "").replace("<|answer_end|>", "").strip()                
            else:
                analysis_text = ""
                answer_text = generated_text
            logger.info("Analysis_text:")
            logger.info(analysis_text)
            logger.info("Answer_text:")
            logger.info(answer_text)
            
            return analysis_text, answer_text
    
        
    def display_format_references(self, text, search_results):
        ref_pattern = r'<ref name="([^"]+)">"([^"]+)"</ref>\.\s*'
        current_pos = 0
        
        seen_hashes = {}
        hash_count = 0
        
        for match in re.finditer(ref_pattern, text):
            # Add text before the reference
            text_before = text[current_pos:match.start()].rstrip()                        
            # Extract reference components
            current_hash = match.group(1)
            
            # Check that the hash is in the search results, otherwise it was hallucinated
            try:
                table_extract = search_results.loc[search_results["hash"] == current_hash]
                if len(table_extract) == 0:
                    continue
                else:
                    ref_url = table_extract["url"].values[0]
                    ref_text_from_db = table_extract["text"].values[0]
            except KeyError:
                logger.error(f"Hash {current_hash} not found in search results.")
                
            
            # Check if the hash has already been quoted
            if current_hash not in seen_hashes.keys():
                hash_count += 1
                hash_number = hash_count
                seen_hashes[current_hash] = {}
                seen_hashes[current_hash]["hash_number"] = hash_number
                seen_hashes[current_hash]["url"] = ref_url
                seen_hashes[current_hash]["text"] = ref_text_from_db
                
            else:
                hash_number = seen_hashes[current_hash]["hash_number"]
            
            quoted_text = match.group(2).strip()
            
            # clickable link with a tooltip on top
            ui.html( 
                f"""
                {text_before}
                <span class="tooltip fade" data-title="{quoted_text}"> 
                    <a href={ref_url} target="_blank">
                        [{hash_number}]
                    </a>
                </span>.<br>
                """
                )

            
            current_pos = match.end()
        ui.label(text[current_pos:])
        
        return seen_hashes
    
    def display_sources(self, seen_hashes):
        for hash in seen_hashes:
            ui.html( 
                f"""
                <a href={seen_hashes[hash]["url"]} target="_blank">
                    [{seen_hashes[hash]["hash_number"]}]
                </a>
                {seen_hashes[hash]["text"]}.<br>
                """
                )


############################
# MAIN UI
############################

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

# HEADER with Tabs
with ui.header(elevated=True).classes('justify-between w-full p-4 bg-[#B63838]') as header:
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


# LIBRARY TAB CONTENT (hidden by default)
with ui.column().classes('w-full hidden') as library_content:
    ui.label('Library Section - Work in progress...')

# CONVERSATIONS TAB CONTENT (hidden by default)
with ui.column().classes('w-full hidden') as conversations_content:
    ui.label('Conversations Section - Work in progress...')

# MAIN TAB CONTENT
with ui.column().classes('w-full max-w-2xl mx-auto items-stretch') as chat_display:
    pass

with ui.footer().classes('bg-white') as query_bar:
    chat_windows = ChatWindows()
    with ui.row().classes('w-full max-w-3xl mx-auto my-2 no-wrap items-center'):
        with ui.avatar():
            ui.image(chat_windows.user_avatar).classes(f'rounded-full bg-[{main_colour}]')
        input_field = ui.input(placeholder="Type your question...") \
            .props('rounded outlined input-class=mx-3').classes('flex-grow') \
            .on('keydown.enter', chat_windows.send_message)
        ui.button("Send", on_click=chat_windows.send_message)
        
    
with ui.right_drawer(fixed=False, value=False).style(f'background-color: {pale_colour}') as analysis_drawer:
    with ui.column() as analysis_container:
        ui.label('Ask a question to see the sources and analysis')
        
analysis_drawer.bind_visibility_from(chat_display)
query_bar.bind_visibility_from(chat_display)


ui.run(host="0.0.0.0", port=8081)
