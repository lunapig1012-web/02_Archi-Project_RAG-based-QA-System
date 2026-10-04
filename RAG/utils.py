#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import Dict, List, Optional, Tuple, Union

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
enc = tiktoken.get_encoding("cl100k_base")


class ReadFiles:
    """
    class to read files
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self.file_list = self.get_files()

    def get_files(self):
        # dir_path：対象フォルダのパス
        file_list = []
        for filepath, dirnames, filenames in os.walk(self._path):
            # os.walk で指定フォルダを再帰的に走査する
            for filename in filenames:
                # 拡張子から対象ファイルかどうかを判定する
                if filename.endswith(".md"):
                    # 対象ファイルの絶対パスを結果リストに追加する
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".txt"):
                    file_list.append(os.path.join(filepath, filename))
                elif filename.endswith(".pdf"):
                    file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_content(self, max_token_len: int = 600, cover_content: int = 150):
        docs = []
        # ファイル内容を読み込む
        for file in self.file_list:
            content = self.read_file_content(file)
            chunk_content = self.get_chunk(
                content, max_token_len=max_token_len, cover_content=cover_content)
            docs.extend(chunk_content)
        return docs

    @classmethod
    def get_chunk(cls, text: str, max_token_len: int = 600, cover_content: int = 150):
        def split_by_token_limit(content: str, token_limit: int):
            chunks = []
            curr_len = 0
            curr_chunk = ''

            for line in content.split('\n'):
                line_len = len(enc.encode(line))
                if line_len > token_limit:
                    print('warning line_len = ', line_len)
                if curr_len + line_len <= token_limit:
                    curr_chunk += line
                    curr_chunk += '\n'
                    curr_len += line_len
                    curr_len += 1
                else:
                    if curr_chunk:
                        chunks.append(curr_chunk)
                    curr_chunk = curr_chunk[-cover_content:]+line
                    curr_len = line_len + cover_content

            if curr_chunk:
                chunks.append(curr_chunk)

            return chunks

        lines = text.split('\n')

        def is_section_heading(line: str) -> bool:
            stripped_line = line.strip()
            return (stripped_line.startswith('【')
                    and stripped_line.endswith('】')
                    and len(stripped_line) > 2)

        def get_heading_title(heading: str) -> str:
            return heading.strip()[1:-1]

        def is_district_heading(heading: str) -> bool:
            return get_heading_title(heading).endswith('地区')

        def parse_document_metadata(source_lines: List[str]):
            metadata = {}
            for source_line in source_lines:
                if is_section_heading(source_line):
                    break

                stripped_line = source_line.strip()
                if '：' not in stripped_line:
                    continue

                key, value = stripped_line.split('：', 1)
                key = key.strip()
                value = value.strip()
                if key and value:
                    metadata[key] = value

            return metadata

        if not any(is_section_heading(line) for line in lines):
            return split_by_token_limit(text, max_token_len)

        sections = []
        current_heading = None
        current_body_lines = []

        for line in lines:
            if is_section_heading(line):
                if current_heading is not None or current_body_lines:
                    sections.append((current_heading, current_body_lines))
                current_heading = line
                current_body_lines = []
            else:
                current_body_lines.append(line)

        if current_heading is not None or current_body_lines:
            sections.append((current_heading, current_body_lines))

        chunk_text = []
        document_metadata = parse_document_metadata(lines)
        document_context_lines = []
        for metadata_key in ('資料名', '地区計画番号'):
            metadata_value = document_metadata.get(metadata_key)
            if metadata_value:
                document_context_lines.append(
                    f'{metadata_key}：{metadata_value}')

        current_district = None

        for heading, body_lines in sections:
            body = '\n'.join(body_lines).strip('\n')
            if heading is None:
                section_text = body
            else:
                heading_title = get_heading_title(heading)
                if is_district_heading(heading):
                    current_district = heading_title
                    context_lines = []
                else:
                    context_lines = document_context_lines.copy()
                    if current_district:
                        context_lines.append(f'地区：{current_district}')
                    context_lines.append(f'項目：{heading_title}')

                section_header = heading
                if context_lines:
                    context_text = '\n'.join(context_lines)
                    section_header = f'{context_text}\n\n{heading}'

                if body:
                    section_text = f'{section_header}\n{body}'
                else:
                    section_text = section_header

            if not section_text:
                continue

            if len(enc.encode(section_text)) <= max_token_len:
                chunk_text.append(section_text)
                continue

            if heading is None:
                chunk_text.extend(
                    split_by_token_limit(section_text, max_token_len))
                continue

            section_header_token_len = len(enc.encode(section_header)) + 1
            body_token_limit = max_token_len - section_header_token_len

            if body_token_limit <= 0:
                if not body:
                    chunk_text.append(section_header)
                    continue

                body_chunks = split_by_token_limit(body, 1)
                for body_chunk in body_chunks:
                    body_chunk = body_chunk.rstrip('\n')
                    if body_chunk:
                        chunk_text.append(f'{section_header}\n{body_chunk}')
                    else:
                        chunk_text.append(section_header)
                continue

            body_chunks = split_by_token_limit(body, body_token_limit)
            for body_chunk in body_chunks:
                body_chunk = body_chunk.rstrip('\n')
                if body_chunk:
                    chunk_text.append(f'{section_header}\n{body_chunk}')
                else:
                    chunk_text.append(section_header)

        return chunk_text

    @classmethod
    def read_file_content(cls, file_path: str):
        # 拡張子に応じて読み込み方法を切り替える
        if file_path.endswith('.pdf'):
            return cls.read_pdf(file_path)
        elif file_path.endswith('.md'):
            return cls.read_markdown(file_path)
        elif file_path.endswith('.txt'):
            return cls.read_text(file_path)
        else:
            raise ValueError("Unsupported file type")

    @classmethod
    def read_pdf(cls, file_path: str):
        # PDFファイルを読み込む
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            return text

    @classmethod
    def read_markdown(cls, file_path: str):
        # Markdownファイルを読み込む
        with open(file_path, 'r', encoding='utf-8') as file:
            md_text = file.read()
            html_text = markdown.markdown(md_text)
            # HTMLからプレーンテキストを抽出する
            text_maker = html2text.HTML2Text()
            text_maker.ignore_links = True
            text = text_maker.handle(html_text)
            return text

    @classmethod
    def read_text(cls, file_path: str):
        # テキストファイルを読み込む
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()


class Documents:
    """
        分類済みのJSON形式ドキュメントを取得する。
    """
    def __init__(self, path: str = '') -> None:
        self.path = path
    
    def get_content(self):
        with open(self.path, mode='r', encoding='utf-8') as f:
            content = json.load(f)
        return content
