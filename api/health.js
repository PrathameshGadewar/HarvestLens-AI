export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  return res.status(200).json({
    status: 'healthy',
    model_ready: true,
    engine: 'HarvestLens AI Cloud Serverless',
    classes_count: 32,
    classes: {
      0: "Apple Overripe", 1: "Apple Ripe", 2: "Apple Rotten", 3: "Apple Unripe",
      4: "Banana Overripe", 5: "Banana Ripe", 6: "Banana Rotten", 7: "Banana Unripe",
      8: "Grape Overripe", 9: "Grape Ripe", 10: "Grape Rotten", 11: "Grape Unripe",
      12: "Mango Overripe", 13: "Mango Ripe", 14: "Mango Rotten", 15: "Mango Unripe",
      16: "Melon Overripe", 17: "Melon Ripe", 18: "Melon Rotten", 19: "Melon Unripe",
      20: "Orange Overripe", 21: "Orange Ripe", 22: "Orange Rotten", 23: "Orange Unripe",
      24: "Peach Overripe", 25: "Peach Ripe", 26: "Peach Rotten", 27: "Peach Unripe",
      28: "Pear Overripe", 29: "Pear Ripe", 30: "Pear Rotten", 31: "Pear Unripe"
    }
  });
}
