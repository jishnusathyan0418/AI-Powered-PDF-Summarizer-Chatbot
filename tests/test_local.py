"""Local integration checks. The LLM is mocked; PDF parsing and retrieval are real.
Run: .venv\Scripts\python.exe -m unittest discover -s tests -v
"""
import io
import os
import unittest
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import server
import worker


def make_pdf(text=None):
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    if text:
        font = DictionaryObject({
            NameObject('/Type'): NameObject('/Font'),
            NameObject('/Subtype'): NameObject('/Type1'),
            NameObject('/BaseFont'): NameObject('/Helvetica'),
        })
        page[NameObject('/Resources')] = DictionaryObject({
            NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})
        })
        escaped = text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')
        stream = DecodedStreamObject()
        stream.set_data(('BT /F1 12 Tf 50 700 Td (' + escaped + ') Tj ET').encode('ascii'))
        page[NameObject('/Contents')] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


class PdfChatbotTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        worker.reset_document()

    def tearDown(self):
        worker.reset_document()

    def upload(self, data, name='handbook.pdf'):
        return self.client.post('/process-document', data={'file': (io.BytesIO(data), name)})

    def test_home_and_assets(self):
        self.assertEqual(self.client.get('/').status_code, 200)
        with self.client.get('/static/script.js') as response:
            self.assertEqual(response.status_code, 200)

    def test_question_before_upload(self):
        response = self.client.post('/process-message', json={'userMessage': 'Hello'})
        self.assertEqual(response.status_code, 400)
        self.assertIn('upload', response.json['botResponse'])

    def test_invalid_question_payloads(self):
        for payload in ({}, [], {'userMessage': ''}, {'userMessage': 42}):
            with self.subTest(payload=payload):
                response = self.client.post('/process-message', json=payload)
                self.assertEqual(response.status_code, 400)
                self.assertIn('botResponse', response.json)

    def test_invalid_uploads(self):
        self.assertEqual(self.client.post('/process-document').status_code, 400)
        self.assertEqual(self.upload(b'not a PDF').status_code, 400)
        self.assertEqual(self.upload(b'text', 'notes.txt').status_code, 400)
        self.assertEqual(self.upload(b'%PDF-1.4\nbroken').status_code, 422)

    def test_empty_pdf(self):
        response = self.upload(make_pdf())
        self.assertEqual(response.status_code, 400)
        self.assertIn('No readable text', response.json['botResponse'])

    def test_size_limit(self):
        with patch.dict(server.app.config, {'MAX_CONTENT_LENGTH': 100}):
            response = self.upload(b'x' * 200)
            self.assertEqual(response.status_code, 413)
            self.assertIn('botResponse', response.json)

    def test_real_pdf_embeddings_retrieval_and_mocked_answer(self):
        content = 'Northstar employees receive 23 days of annual leave. The project codename is ORCHID-742.'
        response = self.upload(make_pdf(content))
        self.assertEqual(response.status_code, 200, response.json)
        retrieved = worker.db.as_retriever(search_type='mmr', search_kwargs={'k': 6}).invoke('What is the vacation allowance?')
        self.assertTrue(any('23 days' in doc.page_content for doc in retrieved))
        model = FakeListChatModel(responses=['Employees receive 23 days of annual leave.'])
        with patch.object(worker, 'init_llm', return_value=model):
            response = self.client.post('/process-message', json={'userMessage': 'How much annual leave?'})
            self.assertEqual(response.status_code, 200)
            self.assertIn('23 days', response.json['botResponse'])
            captured = worker.conversation_retrieval_chain.invoke({'input': 'What is the project codename?'})
            self.assertTrue(any('ORCHID-742' in doc.page_content for doc in captured['context']))
        self.assertEqual(len(worker.chat_history), 1)
        response = self.client.post('/reset')
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(worker.db)
        self.assertEqual(worker.chat_history, [])
        response = self.client.post('/process-message', json={'userMessage': 'Annual leave?'})
        self.assertEqual(response.status_code, 400)

    def test_replacement_pdf_does_not_mix_documents(self):
        self.assertEqual(self.upload(make_pdf('Secret first document code ALPHA-123.')).status_code, 200)
        self.assertEqual(self.upload(make_pdf('Second document code BETA-987.')).status_code, 200)
        chunks = worker.db.get()['documents']
        self.assertTrue(any('BETA-987' in text for text in chunks))
        self.assertFalse(any('ALPHA-123' in text for text in chunks))

    def test_missing_key_returns_clear_error(self):
        self.assertEqual(self.upload(make_pdf('Example PDF for missing key test.')).status_code, 200)
        with patch.object(worker, 'llm_hub', None), patch.object(worker, 'load_dotenv'), patch.dict(os.environ, {'GROQ_API_KEY': ''}):
            response = self.client.post('/process-message', json={'userMessage': 'Summarize this.'})
            self.assertEqual(response.status_code, 503)
            self.assertIn('GROQ_API_KEY', response.json['botResponse'])

    def test_model_failure_is_json(self):
        self.assertEqual(self.upload(make_pdf('Example document.')).status_code, 200)
        with patch.object(worker, 'init_llm', side_effect=ConnectionError('unavailable')):
            response = self.client.post('/process-message', json={'userMessage': 'Summarize this.'})
            self.assertEqual(response.status_code, 502)
            self.assertIn('botResponse', response.json)


if __name__ == '__main__':
    unittest.main()
