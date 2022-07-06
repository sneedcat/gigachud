import asyncio
import base64
from heapq import merge
import html
import os
import random
import re
import time
from typing import List, Optional, Tuple, Union
from dotenv import load_dotenv
import httpx
from telethon import TelegramClient, events
import logging

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s:%(message)s", level=logging.INFO
)
load_dotenv()

WELCOME = (
    "**This is The First Gigachud Bot**\n"
    + "Things it can do:\n"
    + "- download a comic with /xkcd.\n"
    + "- download videos from youtube with /yt <url>.\n"
    + "- generate random images with /dalle <prompt>.\n"
    + "- generate random text with /gpt <prompt>.\n"
    + "- generate tts using tiktok tts with /tts <text>."
)

tg_api_id = os.getenv("TG_API_ID")
tg_api_hash = os.getenv("TG_API_HASH")
bot_token = os.getenv("BOT_TOKEN")


class Dalle:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.client = httpx.AsyncClient()

    async def generate(self) -> List[bytes]:
        logging.info("Generating images with prompt `{}`".format(self.prompt))
        for i in range(0, 3):
            try:
                result = await self.request_images()
                return result
            except Exception as e:
                logging.warning(e)
        raise Exception("Unknown error")

    async def request_images(self) -> List[bytes]:
        body = dict(prompt=self.prompt)
        url = "https://bf.dallemini.ai/generate"
        try:
            response = await self.client.post(url, json=body, timeout=180)
        except:
            raise Exception("Request timeout out")
        if response.status_code != 200:
            raise Exception(
                "Status code isn't 200, it is {}".format(response.status_code)
            )
        data = response.json()
        images: List[str] = data["images"]
        parsed_images = []
        for image in images:
            parsed_images.append(base64.b64decode(image))
        return parsed_images


class GPTJ:
    def __init__(self, prompt: str):
        self.prompt = prompt
        self.client = httpx.AsyncClient()

    async def generate(self) -> str:
        logging.info("Generating text with prompt `{}`".format(self.prompt))
        for _i in range(0, 3):
            try:
                return await self.request_text()
            except Exception as e:
                logging.warn(e)
                await asyncio.sleep(5)
        raise Exception("Unknown error")

    async def request_text(self) -> str:
        try:
            resp = await self.client.get("https://textsynth.com/playground.html")
            regex = re.compile('var textsynth_api_key = "(.*?)"')
            l = regex.findall(resp.text)
            if len(l) == 0:
                raise Exception("Internal error.")
            api_key = l[0]
        except:
            raise Exception("Request timeout out")
        body = dict(
            prompt=self.prompt,
            temperature=0.95,
            stream=False,
            top_k=40,
            top_p=0.9,
            max_tokens=200,
            stop=None,
        )
        url = "https://api.textsynth.com/v1/engines/gptj_6B/completions"
        headers = {
            "Authorization": "Bearer {}".format(api_key),
        }
        try:
            response = await self.client.post(
                url, json=body, timeout=10, headers=headers
            )
        except:
            raise Exception("Request timeout out")
        if response.status_code != 200:
            raise Exception(
                "Status code isn't 200, it is {}".format(response.status_code)
            )
        data = response.json()
        return f"**{self.prompt}** {data['text']}"


class Ytdlp:
    def __init__(self, url: str):
        self.url = url
        self.client = httpx.AsyncClient()

    async def download(self) -> str:
        logging.info("Downloading video with url `{}`".format(self.url))
        for i in range(0, 3):
            try:
                return await self.request_video()
            except Exception as e:
                logging.warning(e)
                await asyncio.sleep(5)
        raise Exception("Unknown error")

    async def request_video(self) -> str:
        name = str(random.randint(100000000000, 999999999999)) + ".mp4"
        proc = await asyncio.create_subprocess_shell(
            'yt-dlp -f "[filesize<100M]" --merge-output-format mp4 -o "{}" "{}"'.format(
                name, self.url
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            logging.warning(stderr)
            raise Exception(
                "Process returned non-zero exit code: {}".format(proc.returncode)
            )
        return name


class Xkcd:
    def __init__(self, index: Optional[str]):
        self.client = httpx.AsyncClient()
        self.index = index

    async def generate(self) -> Tuple[str, str]:
        if not self.index:
            for _ in range(0, 3):
                try:
                    resp = await self.client.get("https://xkcd.com/")
                    if resp.status_code != 200:
                        resp = None
                        continue
                except:
                    resp = None
                    continue
            if resp is None:
                raise Exception("Couldn't get xkcd index")

            regex = re.compile('comic: <a href="https://xkcd.com/(.*?)">')
            l = regex.findall(resp.text)
            if len(l) == 0:
                raise Exception("Internal error.")
            r = random.randint(1, int(l[0]) + 1)
            if r == 404:
                r = random.randint(1, 404)
            self.index = r
        else:
            self.index = int(self.index)
        logging.info("Downloading xkcd comic with index `{}`".format(self.index))
        for _i in range(0, 3):
            try:
                return await self.request_comic()
            except Exception as e:
                logging.warning(e)
                await asyncio.sleep(5)
        raise Exception("Couldn't get a xkcd comic")

    async def request_comic(self) -> Tuple[str, str]:
        url = "https://xkcd.com/{}/".format(self.index)
        try:
            resp = await self.client.get(url, timeout=10)
            regex = re.compile(
                '<div id="comic">\n<img src="(.+?)" title=".+?" alt="(.+?)"'
            )
            l = regex.findall(resp.text)
            if len(l) == 0:
                raise Exception("Internal error. Regex couldn't find the link")
            image_url = l[0][0]
            image_title = l[0][1]
            return (image_url, image_title)
        except Exception as e:
            raise Exception(e)


class TiktokTTS:
    def __init__(self, prompt: str):
        self.client = httpx.AsyncClient()
        self.prompt = prompt
        filename = str(random.randint(1000000000000, 6000000000000000000))
        self.filename = filename

    async def generate(self):
        voice = "en_us_001"
        for _i in range(0, 3):
            try:
                return await self.request_tts(voice)
            except Exception as e:
                logging.warning(e)
                await asyncio.sleep(5)
        raise Exception("Couldn't create the tts file")
    
    def sanitize_text(self, text: str) -> str:
        r"""Sanitizes the text for tts.
            What gets removed:
        - following characters`^_~@!&;#:-%“”‘"%*/{}[]()\|<>?=+`
        - any http or https links
        Args:
            text (str): Text to be sanitized
        Returns:
            str: Sanitized text
        """

        # remove any urls from the text
        regex_urls = r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*"

        result = re.sub(regex_urls, " ", text)

        # note: not removing apostrophes
        regex_expr = r"\s['|’]|['|’]\s|[\^_~@!&;#:\-%“”‘\"%\*/{}\[\]\(\)\\|<>=+]"
        result = re.sub(regex_expr, " ", result)
        result = result.replace("+", "plus").replace("&", "and")
        # remove extra whitespace
        return " ".join(result.split())

    async def request_tts(self, voice: str):
        prompt = self.sanitize_text(self.prompt)
        frames = 0
        while len(prompt) > 300:
            index = 300
            while index >= 0 and (self.prompt[index] not in " \n,\t\r,'"):
                index -= 1
            index += 1
            current_prompt = prompt[0:index]
            prompt = prompt[index:]
            URI_BASE = "https://api16-normal-useast5.us.tiktokv.com/media/api/text/speech/invoke/?text_speaker="
            try:
                resp = await self.client.post(
                    f"{URI_BASE}{voice}&req_text={current_prompt}&speaker_map_type=0"
                )
                vstr = [resp.json()["data"]["v_str"]][0]
                b64 = base64.b64decode(vstr)
                filename = f"{self.filename}_{frames}.mp3"
                with open(filename, "wb") as file:
                    file.write(b64)
                frames += 1
            except:
                logging.error("Couldn't get current tts")
        if len(prompt) != 0:
            URI_BASE = "https://api16-normal-useast5.us.tiktokv.com/media/api/text/speech/invoke/?text_speaker="
            try:
                resp = await self.client.post(
                    f"{URI_BASE}{voice}&req_text={prompt}&speaker_map_type=0"
                )
                vstr = [resp.json()["data"]["v_str"]][0]
                b64 = base64.b64decode(vstr)
                filename = f"{self.filename}_{frames}.mp3"
                with open(filename, "wb") as file:
                    file.write(b64)
                frames += 1
            except:
                logging.error("Couldn't get current tts")
        merge_file = ""
        for i in range(frames):
            merge_file += f"file '{self.filename}_{i}.mp3'\n"
        merge_file_name = str(random.randint(4124124124, 41241241241241)) + ".txt"
        with open(merge_file_name, "w") as file:
            file.write(merge_file)
        command = f"ffmpeg -f ffmpeg -f concat -i {merge_file_name} -c copy {self.filename}.mp3"
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        for i in range(frames):
            os.remove(f"{self.filename}_{i}.mp3")
        os.remove(merge_file_name)
        if proc.returncode != 0:
            logging.warning(stderr)
            raise Exception(
                "Process returned non-zero exit code: {}".format(proc.returncode)
            )


previous = time.time()


@events.register(events.MessageEdited)
async def test(update):
    print(update.original_update)
    print(update.message)


@events.register(events.NewMessage)
async def handler(event):
    global previous
    now = time.time()
    if now > previous + 50:
        logging.info("It's stil running")
        previous = now
    event: events.NewMessage.Event = event
    if event.message.text.startswith("/help") or event.message.text.startswith(
        "/start"
    ):
        await client.send_message(event.chat_id, WELCOME, parse_mode="md")
    elif event.message.text.startswith("/dalle"):
        parts = event.message.text.split(" ")
        if len(parts) == 1:
            await event.respond("Please provide a prompt")
            return
        prompt = " ".join(parts[1:])
        dalle = Dalle(prompt)
        try:
            result = await dalle.generate()
            await client.send_file(
                event.chat_id, result, caption="Caption: {}".format(prompt)
            )
        except Exception as e:
            logging.error(e)
            await event.respond(
                "Dalle mini is not working right now, please try again later"
            )
    elif event.message.text.startswith("/gpt"):
        parts = event.message.text.split(" ")
        if len(parts) == 1:
            await event.respond("Please provide a prompt")
            return
        prompt = " ".join(parts[1:])
        gptj = GPTJ(prompt)
        try:
            result = await gptj.generate()
            await client.send_message(
                event.chat_id, message=result, reply_to=event.message.id
            )
        except Exception as e:
            logging.error(e)
            await event.respond(
                "GPT-J is not working right now, please try again later"
            )

    elif event.message.text.startswith("/yt"):
        parts = event.message.text.split(" ")
        if len(parts) == 1:
            await event.respond("Please provide a url")
            return
        url = parts[1]
        loading = client.send_file(
            event.chat_id, "./loading-load.gif", reply_to=event.message.id
        )
        ytdlp = Ytdlp(url)
        try:
            result = ytdlp.download()
            loading = await loading
            result = await result
            await client.delete_messages(event.chat_id, loading.id)
            await client.send_file(event.chat_id, result, reply_to=event.message.id)
            os.remove(result)
        except Exception as e:
            logging.error(e)
            await client.delete_messages(event.chat_id, loading.id)
            await event.respond(
                "YT-DLP is not working right now, please try again later"
            )
    elif event.message.text.startswith("/xkcd"):
        parts = event.message.text.split(" ")
        index = parts[1] if len(parts) > 1 else None
        xkcd = Xkcd(index)
        try:
            image = await xkcd.generate()
            if len(image[1]) > 100:
                caption = image[1]
            else:
                caption = image[1]

            await client.send_file(
                event.chat_id, "https:" + image[0], caption="{}".format(caption)
            )
        except Exception as e:
            logging.error(e)
            await event.respond("XKCD is not working right now, please try again later")
    elif event.message.text.startswith("/tts"):
        parts = event.message.text.split(" ", 1)
        if len(parts) == 1:
            await client.send_message(event.chat_id, "Please provide a prompt")
        tts = TiktokTTS(parts[1])
        try:
            await tts.generate()
            if len(tts.prompt) > 100:
                caption = tts.prompt[:100]
            else:
                caption = tts.prompt
            filename = tts.filename + ".mp3"
            await client.send_file(
                event.chat_id,
                filename,
                voice_note=True,
                caption="Caption: {}".format(caption),
            )
            os.remove(filename)
        except Exception as e:
            logging.error(e)
            await event.respond(
                "Tiktok TTS is not working right now, please try again later"
            )


client = TelegramClient("gigachud_tg", tg_api_id, tg_api_hash).start(
    bot_token=bot_token
)

client.add_event_handler(handler)
client.add_event_handler(test)

print("(Press Ctrl+C to stop this)")
try:
    client.run_until_disconnected()
finally:
    client.disconnect()
