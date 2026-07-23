import unittest


class ConfigTests(unittest.TestCase):
    def test_multi_unseen_override_partitions_all_classes(self):
        from phyrc_gzsl.utils.config import set_unseen_classes

        config = {"classes": {"names": {1: "a", 2: "b", 3: "c", 4: "d"}}}
        set_unseen_classes(config, [2, 4])

        self.assertEqual(config["classes"]["seen_classes"], [1, 3])
        self.assertEqual(config["classes"]["unseen_classes"], [2, 4])

    def test_multi_unseen_override_rejects_invalid_sets(self):
        from phyrc_gzsl.utils.config import set_unseen_classes

        config = {"classes": {"names": {1: "a", 2: "b", 3: "c"}}}
        for unseen, message in (([], "non-empty"), ([2, 2], "unique"), ([4], "Unknown")):
            with self.subTest(unseen=unseen), self.assertRaisesRegex(ValueError, message):
                set_unseen_classes(config, unseen)

    def test_single_unseen_override_partitions_all_classes(self):
        from phyrc_gzsl.utils.config import set_single_unseen_class

        config = {"classes": {"names": {1: "a", 2: "b", 3: "c"}}}
        set_single_unseen_class(config, 2)

        self.assertEqual(config["classes"]["seen_classes"], [1, 3])
        self.assertEqual(config["classes"]["unseen_classes"], [2])

    def test_single_unseen_override_rejects_unknown_class(self):
        from phyrc_gzsl.utils.config import set_single_unseen_class

        with self.assertRaisesRegex(ValueError, "Unknown class id"):
            set_single_unseen_class({"classes": {"names": {1: "a"}}}, 2)


if __name__ == "__main__":
    unittest.main()
