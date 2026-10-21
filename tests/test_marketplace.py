import unittest
import uuid
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.database import Base, engine

client = TestClient(app)


class TestMarketplace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)

        # Farmer
        farmer_res = client.post("/api/auth/demo-login", json={"role": "farmer"})
        cls.farmer_token = farmer_res.json()["access_token"]
        cls.farmer_id = farmer_res.json()["user_id"]

        # Buyer
        buyer_res = client.post("/api/auth/demo-login", json={"role": "buyer"})
        cls.buyer_token = buyer_res.json()["access_token"]
        cls.buyer_id = buyer_res.json()["user_id"]

        # Admin
        admin_res = client.post("/api/auth/demo-login", json={"role": "admin"})
        cls.admin_token = admin_res.json()["access_token"]

    def test_marketplace_flow_and_moderation(self):
        farmer_headers = {"Authorization": f"Bearer {self.farmer_token}"}
        buyer_headers = {"Authorization": f"Bearer {self.buyer_token}"}
        admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 1. Farmer creates a batch
        batch_res = client.post("/api/batches", json={
            "name": "Organic Orange Harvest",
            "fruit": "Orange",
            "variety": "Nagpur Mandarin",
            "plot": "South Grove",
            "interval_days": 7,
            "qty": 1000.0,
            "unit": "kg",
            "confirmed_category": "Ripe"
        }, headers=farmer_headers)
        self.assertEqual(batch_res.status_code, 201)
        batch_id = batch_res.json()["id"]

        # 2. Farmer publishes batch to the marketplace
        publish_payload = {
            "title": "Fresh Nagpur Oranges - Premium Quality",
            "region": "Nagpur / Vidarbha",
            "offered_qty": 600.0,
            "price_per_unit": 75.0,
            "currency": "INR",
            "description": "Sweet, juicy, tree-ripened oranges harvested at peak maturity.",
            "publish_now": True
        }
        pub_res = client.post(f"/api/marketplace/listings/{batch_id}/publish", json=publish_payload, headers=farmer_headers)
        self.assertEqual(pub_res.status_code, 200)
        listing = pub_res.json()
        listing_id = listing["id"]
        self.assertEqual(listing["status"], "published")
        self.assertEqual(listing["available_qty"], 600.0)

        # 3. Buyer searches marketplace
        search_res = client.get("/api/marketplace/listings?fruit=Orange&q=Nagpur")
        self.assertEqual(search_res.status_code, 200)
        listings = search_res.json()
        found = any(l["id"] == listing_id for l in listings)
        self.assertTrue(found)

        # 4. Buyer submits a purchase request
        req_res = client.post("/api/requests", json={
            "listing_id": listing_id,
            "quantity": 200.0,
            "proposed_price": 70.0,
            "message": "Interested in 200 kg for local distribution."
        }, headers=buyer_headers)
        self.assertEqual(req_res.status_code, 201)
        request_id = req_res.json()["id"]
        self.assertEqual(req_res.json()["status"], "Pending")

        # 5. Farmer accepts the purchase request
        accept_res = client.put(f"/api/requests/{request_id}/status", json={"status": "Accepted"}, headers=farmer_headers)
        self.assertEqual(accept_res.status_code, 200)
        self.assertEqual(accept_res.json()["status"], "Accepted")

        # 6. Verify available quantity decreased from 600 to 400
        listing_check = client.get(f"/api/marketplace/listings/{listing_id}")
        self.assertEqual(listing_check.status_code, 200)
        self.assertEqual(listing_check.json()["reserved_qty"], 200.0)
        self.assertEqual(listing_check.json()["available_qty"], 400.0)

        # 7. Admin suspends listing
        mod_res = client.post(
            f"/api/admin/listings/{listing_id}/moderate",
            json={"suspend": True, "reason": "Verifying seller certification"},
            headers=admin_headers
        )
        self.assertEqual(mod_res.status_code, 200)
        self.assertTrue(mod_res.json()["is_suspended"])

        # 8. Suspended listing must NOT appear in public buyer search
        search_after = client.get("/api/marketplace/listings?fruit=Orange")
        self.assertFalse(any(l["id"] == listing_id for l in search_after.json()))

        # 9. Verify audit log was recorded
        audit_res = client.get("/api/admin/audit-logs", headers=admin_headers)
        self.assertEqual(audit_res.status_code, 200)
        logs = audit_res.json()
        self.assertTrue(any(l["target_id"] == str(listing_id) and l["action"] == "Suspended listing" for l in logs))

        # 10. Admin restores listing
        restore_res = client.post(
            f"/api/admin/listings/{listing_id}/moderate",
            json={"suspend": False, "reason": "Certification verified and approved"},
            headers=admin_headers
        )
        self.assertEqual(restore_res.status_code, 200)
        self.assertFalse(restore_res.json()["is_suspended"])


if __name__ == "__main__":
    unittest.main()
