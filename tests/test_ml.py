import io
import unittest
from PIL import Image
from backend.app.ml.detector import FruitDetector, parse_fruit_maturity


class TestMLDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FruitDetector.get_instance()

    def test_model_loaded_and_classes(self):
        self.assertIsNotNone(self.detector.model)
        self.assertIsNotNone(self.detector.names)
        self.assertEqual(len(self.detector.names), 32)
        # Verify all 8 fruits and 4 stages are represented
        classes_str = " ".join(self.detector.names.values())
        for fruit in ["Apple", "Banana", "Grape", "Mango", "Melon", "Orange", "Peach", "Pear"]:
            self.assertIn(fruit, classes_str)
        for stage in ["Unripe", "Ripe", "Overripe", "Rotten"]:
            self.assertIn(stage, classes_str)

    def test_parse_fruit_maturity(self):
        res1 = parse_fruit_maturity("Apple Overripe")
        self.assertEqual(res1["fruit"], "Apple")
        self.assertEqual(res1["maturity"], "Overripe")

        res2 = parse_fruit_maturity("Mango Ripe")
        self.assertEqual(res2["fruit"], "Mango")
        self.assertEqual(res2["maturity"], "Ripe")

        res3 = parse_fruit_maturity("Banana_Rotten")
        self.assertEqual(res3["fruit"], "Banana")
        self.assertEqual(res3["maturity"], "Rotten")

    def test_inference_on_synthetic_image(self):
        # Create a synthetic fruit-like image
        img = Image.new("RGB", (640, 640), color=(110, 160, 60))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        # Run real inference
        pred = self.detector.predict(image_bytes, confidence_threshold=0.2)
        self.assertIn("detections", pred)
        self.assertIn("image_width", pred)
        self.assertIn("image_height", pred)
        self.assertIn("inference_time_ms", pred)
        self.assertEqual(pred["image_width"], 640)
        self.assertEqual(pred["image_height"], 640)
        self.assertGreater(pred["inference_time_ms"], 0)

        # Check detection structure if any detected
        for d in pred["detections"]:
            self.assertIn("label", d)
            self.assertIn("fruit", d)
            self.assertIn("maturity", d)
            self.assertIn("confidence", d)
            self.assertIn("box", d)
            self.assertEqual(len(d["box"]), 4)

    def test_dominant_maturity_calculation(self):
        sample_dets = [
            {"fruit": "Mango", "maturity": "Ripe", "confidence": 0.88},
            {"fruit": "Mango", "maturity": "Ripe", "confidence": 0.92},
            {"fruit": "Mango", "maturity": "Unripe", "confidence": 0.45}
        ]
        dom = FruitDetector.calculate_dominant_maturity(sample_dets)
        self.assertEqual(dom, "Ripe")


if __name__ == "__main__":
    unittest.main()
