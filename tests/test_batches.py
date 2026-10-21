import unittest
import uuid
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import Base, engine

client = TestClient(app)


class TestBatches(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

        # Create two distinct farmers
        uid1 = uuid.uuid4().hex[:6]
        uid2 = uuid.uuid4().hex[:6]

        res1 = client.post("/api/auth/register", json={
            "email": f"farmer1_{uid1}@test.com",
            "username": f"farmer1_{uid1}",
            "password": "password123",
            "full_name": "Farmer One",
            "role": "farmer",
            "farm_name": "Farm Alpha"
        })
        cls.token1 = res1.json()["access_token"]

        res2 = client.post("/api/auth/register", json={
            "email": f"farmer2_{uid2}@test.com",
            "username": f"farmer2_{uid2}",
            "password": "password123",
            "full_name": "Farmer Two",
            "role": "farmer",
            "farm_name": "Farm Beta"
        })
        cls.token2 = res2.json()["access_token"]

        # Buyer token
        buyer_res = client.post("/api/auth/demo-login", json={"role": "buyer"})
        cls.buyer_token = buyer_res.json()["access_token"]

    def test_create_and_manage_batch(self):
        headers = {"Authorization": f"Bearer {self.token1}"}
        payload = {
            "name": "Apple Crisp Batch A",
            "fruit": "Apple",
            "variety": "Honeycrisp",
            "plot": "West Ridge",
            "interval_days": 4,
            "qty": 500.0,
            "unit": "kg",
            "confirmed_category": "Ripe",
            "notes": "Crisp fruit with rich fragrance."
        }
        res = client.post("/api/batches", json=payload, headers=headers)
        self.assertEqual(res.status_code, 201)
        batch = res.json()
        batch_id = batch["id"]
        self.assertEqual(batch["name"], payload["name"])
        self.assertEqual(batch["fruit"], "Apple")
        self.assertEqual(batch["confirmed_category"], "Ripe")

        # Farmer 1 can view this batch
        detail_res = client.get(f"/api/batches/{batch_id}", headers=headers)
        self.assertEqual(detail_res.status_code, 200)

        # Farmer 2 CANNOT view Farmer 1's private batch (Privacy Enforcement)
        headers2 = {"Authorization": f"Bearer {self.token2}"}
        forbidden_res = client.get(f"/api/batches/{batch_id}", headers=headers2)
        self.assertEqual(forbidden_res.status_code, 403)

        # Buyer CANNOT view private batches
        buyer_headers = {"Authorization": f"Bearer {self.buyer_token}"}
        buyer_forbidden = client.get(f"/api/batches/{batch_id}", headers=buyer_headers)
        self.assertEqual(buyer_forbidden.status_code, 403)

        # Update batch
        update_res = client.put(f"/api/batches/{batch_id}", json={"qty": 650.0, "is_completed": True}, headers=headers)
        self.assertEqual(update_res.status_code, 200)
        self.assertEqual(update_res.json()["qty"], 650.0)
        self.assertTrue(update_res.json()["is_completed"])

    def test_add_observation_with_photo(self):
        headers = {"Authorization": f"Bearer {self.token1}"}
        batch_res = client.post("/api/batches", json={
            "name": "Banana Green Batch",
            "fruit": "Banana",
            "interval_days": 6,
            "qty": 300.0,
            "unit": "kg"
        }, headers=headers)
        batch_id = batch_res.json()["id"]

        # Add observation with synthetic image upload
        import io
        from PIL import Image
        img = Image.new("RGB", (300, 300), color=(80, 180, 80))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        obs_res = client.post(
            f"/api/batches/{batch_id}/observations",
            data={"notes": "Weekly maturity check", "analysed": "true"},
            files={"file": ("photo.jpg", buf, "image/jpeg")},
            headers=headers
        )
        self.assertEqual(obs_res.status_code, 200)
        obs_data = obs_res.json()
        self.assertEqual(obs_data["notes"], "Weekly maturity check")
        self.assertIsNotNone(obs_data["image_url"])


if __name__ == "__main__":
    unittest.main()
