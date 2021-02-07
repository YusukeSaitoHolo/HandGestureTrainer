import unittest
import LoadGestureData

class TestLoadGestureData(unittest.TestCase):
 
  def test_load_json_data(self):
        filepath = '../GestureData/bloom/bloom[2020_4_13-8_45_52_183].json'
        json_data = LoadGestureData.load_json_data(filepath)

        value_frameCount = json_data['gestureData'][1]['frameCount']
        self.assertEqual(1, value_frameCount) 

  def test_generate_labeldata(self):
        filepath = '../GestureData/bloom/bloom[2020_4_13-8_45_52_183].json'
        json_data = LoadGestureData.load_json_data(filepath)
        gesture_data = LoadGestureData.generate_labeldata(json_data)

        self.assertEqual(120, len(gesture_data)) # 次元数が合う
        self.assertEqual(0.000724124547559768, gesture_data[0]) # 最初の数値が合う
        self.assertEqual(0.27374643087387085, gesture_data[-1]) # 最後の数値が合う
 

if __name__ == "__main__":
    unittest.main()