import json
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from rsi_library.server import LocalServer


class HttpApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.server = LocalServer(0)
        except PermissionError as exc:
            raise unittest.SkipTest("This environment does not allow binding a local HTTP port.") from exc
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def fetch(self, path, body=None, **headers):
        request = Request(self.server.url + path, body, headers)
        return urlopen(request, timeout=5)

    def test_workbench_and_api_use_the_same_configs(self):
        for path, media in [("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css")]:
            with self.fetch(path) as response:
                self.assertEqual(response.headers.get_content_type(), media)
                self.assertGreater(len(response.read()), 100)
        with self.fetch("/api/catalog") as response:
            catalog = json.load(response)
        first, second, *_ = (entry["config"] for entry in catalog["templates"])
        for path, payload in [("/api/run", first), ("/api/compare", {"configs": [first, second]})]:
            with self.fetch(path, json.dumps(payload).encode(), **{"Content-Type": "application/json"}) as response:
                result = json.load(response)
                if path == "/api/compare":
                    self.assertEqual(len(result["runs"]), 2)
                    self.assertEqual(result["runs"][0]["trials"], single["trials"])
                else:
                    single = result
                    self.assertEqual(single["evaluations_used"], first["budget"])

    def test_invalid_experiment_returns_a_readable_error(self):
        with self.assertRaises(HTTPError) as caught:
            self.fetch("/api/run", b'{"budget":true}', **{"Content-Type": "application/json"})
        with caught.exception as response:
            self.assertEqual(response.code, 400)
            self.assertTrue(json.load(response)["error"])
