import unittest
from fastapi.testclient import TestClient
from backend.app.main import app

class TestMessagingAndNotifications(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

        # Login as farmer
        f_resp = self.client.post("/api/auth/demo-login", json={"role": "farmer"})
        self.assertEqual(f_resp.status_code, 200)
        self.farmer_token = f_resp.json()["access_token"]
        self.farmer_id = f_resp.json()["user_id"]
        self.farmer_headers = {"Authorization": f"Bearer {self.farmer_token}"}

        # Login as buyer
        b_resp = self.client.post("/api/auth/demo-login", json={"role": "buyer"})
        self.assertEqual(b_resp.status_code, 200)
        self.buyer_token = b_resp.json()["access_token"]
        self.buyer_id = b_resp.json()["user_id"]
        self.buyer_headers = {"Authorization": f"Bearer {self.buyer_token}"}

    def test_bidirectional_messaging_and_notifications(self):
        # 1. Buyer sends message to farmer
        msg_payload = {
            "receiver_id": self.farmer_id,
            "content": "Hello Farmer! Are the Alphonso mangoes ready for shipment this week?"
        }
        res = self.client.post("/api/messages", json=msg_payload, headers=self.buyer_headers)
        self.assertEqual(res.status_code, 201)
        msg_id = res.json()["id"]

        # 2. Verify Farmer receives it in inbox
        farmer_inbox = self.client.get("/api/messages/inbox", headers=self.farmer_headers)
        self.assertEqual(farmer_inbox.status_code, 200)
        inbox_ids = [m["id"] for m in farmer_inbox.json()]
        self.assertIn(msg_id, inbox_ids)

        # 3. Verify Farmer unread count
        farmer_unread = self.client.get("/api/messages/unread-count", headers=self.farmer_headers)
        self.assertEqual(farmer_unread.status_code, 200)
        self.assertGreaterEqual(farmer_unread.json()["unread_messages"], 1)

        # 4. Farmer replies to buyer
        reply_payload = {
            "receiver_id": self.buyer_id,
            "content": "Yes, they were harvested yesterday and are currently in prime ripe condition."
        }
        res_reply = self.client.post("/api/messages", json=reply_payload, headers=self.farmer_headers)
        self.assertEqual(res_reply.status_code, 201)
        reply_id = res_reply.json()["id"]

        # 5. Verify Buyer receives the farmer's reply in inbox
        buyer_inbox = self.client.get("/api/messages/inbox", headers=self.buyer_headers)
        self.assertEqual(buyer_inbox.status_code, 200)
        buyer_inbox_ids = [m["id"] for m in buyer_inbox.json()]
        self.assertIn(reply_id, buyer_inbox_ids)

        # 6. Verify Buyer unread count
        buyer_unread = self.client.get("/api/messages/unread-count", headers=self.buyer_headers)
        self.assertEqual(buyer_unread.status_code, 200)
        self.assertGreaterEqual(buyer_unread.json()["unread_messages"], 1)

        # 7. Check full conversation thread
        conv = self.client.get(f"/api/messages?other_user_id={self.farmer_id}", headers=self.buyer_headers)
        self.assertEqual(conv.status_code, 200)
        conv_ids = [m["id"] for m in conv.json()]
        self.assertIn(msg_id, conv_ids)
        self.assertIn(reply_id, conv_ids)

    def test_proposal_notifications_flow(self):
        # 1. Farmer creates batch and publishes listing
        batch_resp = self.client.post("/api/batches", json={
            "name": "Notification Test Batch",
            "fruit": "Mango",
            "variety": "Alphonso",
            "plot": "Plot North",
            "interval_days": 5,
            "qty": 300,
            "unit": "kg",
            "confirmed_category": "Ripe"
        }, headers=self.farmer_headers)
        self.assertEqual(batch_resp.status_code, 201)
        batch_id = batch_resp.json()["id"]

        pub_resp = self.client.post(f"/api/marketplace/listings/{batch_id}/publish", json={
            "title": "Premium Ratnagiri Alphonso Mangoes",
            "offered_qty": 200,
            "price_per_unit": 150.0,
            "region": "Ratnagiri",
            "farm_name": "Ratnagiri Mango Estate"
        }, headers=self.farmer_headers)
        self.assertEqual(pub_resp.status_code, 200)
        listing_id = pub_resp.json()["id"]

        # 2. Buyer submits purchase proposal with note
        prop_resp = self.client.post("/api/requests", json={
            "listing_id": listing_id,
            "quantity": 50,
            "proposed_price": 140.0,
            "message": "Can we arrange express freight?"
        }, headers=self.buyer_headers)
        self.assertEqual(prop_resp.status_code, 201)
        prop_data = prop_resp.json()
        req_id = prop_data["id"]

        # Verify seller info in response schema
        self.assertEqual(prop_data["farmer_id"], self.farmer_id)
        self.assertEqual(prop_data["seller_name"], "Ratnagiri Mango Estate")

        # 3. Farmer inbox receives notification message automatically
        farmer_inbox = self.client.get("/api/messages/inbox", headers=self.farmer_headers)
        self.assertEqual(farmer_inbox.status_code, 200)
        inbox_msgs = farmer_inbox.json()
        matching_msg = next((m for m in inbox_msgs if m.get("request_id") == req_id), None)
        self.assertIsNotNone(matching_msg, "Farmer inbox must contain automated proposal message")
        self.assertIn("Can we arrange express freight?", matching_msg["content"])

        # 4. Farmer accepts proposal
        status_resp = self.client.put(f"/api/requests/{req_id}/status", json={"status": "Accepted"}, headers=self.farmer_headers)
        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(status_resp.json()["status"], "Accepted")

        # 5. Buyer inbox receives acceptance notification
        buyer_inbox = self.client.get("/api/messages/inbox", headers=self.buyer_headers)
        self.assertEqual(buyer_inbox.status_code, 200)
        buyer_msgs = buyer_inbox.json()
        accept_msg = next((m for m in buyer_msgs if m.get("request_id") == req_id and "ACCEPTED" in m["content"]), None)
        self.assertIsNotNone(accept_msg, "Buyer inbox must contain proposal acceptance message")

if __name__ == "__main__":
    unittest.main()
