"""
UCR Dataset Information Module

This module contains the list of all UCR time series classification datasets
and their domain classifications.
"""

all_ucr_datasets = [
    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX", "CricketY",
    "CricketZ", "Crop", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "DodgerLoopDay", "DodgerLoopGame",
    "DodgerLoopWeekend", "Earthquakes", "ECG200", "ECG5000", "ECGFiveDays", "ElectricDevices",
    "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour",
    "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3",
    "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint", "GunPointAgeSpan", "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung", "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty",
    "InlineSkate", "InsectEPGRegularTrain", "InsectEPGElectricTrain", "InsectEPGMixedTrain",
    "InsectEPGSmallTrain", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances",
    "Lightning2", "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MixedShapesRegularTrain",
    "MixedShapesSmallTrain", "MoteStrain", "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2", "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect",
    "Phoneme", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", "PigCVP",
    "PLAID", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices",
    "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll", "SmallKitchenAppliances",
    "SmoothSubspace", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves",
    "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl", "ToeSegmentation1",
    "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer",
    "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga"
]

dataset_domains = {
    # Sensor data
    "ACSF1": "sensor", "Adiac": "sensor", "AllGestureWiimoteX": "motion", "AllGestureWiimoteY": "motion", "AllGestureWiimoteZ": "motion",
    "ArrowHead": "motion", "Beef": "spectro", "BeetleFly": "spectro", "BirdChicken": "spectro", "BME": "sensor", "Car": "sensor",
    "CBF": "simulated", "Chinatown": "traffic", "ChlorineConcentration": "sensor", "CinCECGTorso": "sensor", "Coffee": "spectro",
    "Computers": "device", "CricketX": "motion", "CricketY": "motion", "CricketZ": "motion", "Crop": "sensor", "DiatomSizeReduction": "image",
    "DistalPhalanxOutlineAgeGroup": "image", "DistalPhalanxOutlineCorrect": "image", "DistalPhalanxTW": "image", "DodgerLoopDay": "traffic",
    "DodgerLoopGame": "traffic", "DodgerLoopWeekend": "traffic", "Earthquakes": "sensor", "ECG200": "sensor", "ECG5000": "sensor",
    "ECGFiveDays": "sensor", "ElectricDevices": "device", "EOGHorizontalSignal": "sensor", "EOGVerticalSignal": "sensor", "EthanolLevel": "sensor",
    "FaceAll": "image", "FaceFour": "image", "FacesUCR": "image", "FiftyWords": "image", "Fish": "image", "FordA": "sensor",
    "FordB": "sensor", "FreezerRegularTrain": "device", "FreezerSmallTrain": "device", "Fungi": "spectro", "GestureMidAirD1": "motion",
    "GestureMidAirD2": "motion", "GestureMidAirD3": "motion", "GesturePebbleZ1": "motion", "GesturePebbleZ2": "motion", "GunPoint": "motion",
    "GunPointAgeSpan": "motion", "GunPointMaleVersusFemale": "motion", "GunPointOldVersusYoung": "motion", "Ham": "spectro", "HandOutlines": "image",
    "Haptics": "sensor", "Herring": "spectro", "HouseTwenty": "device", "InlineSkate": "motion", "InsectEPGRegularTrain": "sensor",
    "InsectEPGElectricTrain": "sensor", "InsectEPGMixedTrain": "sensor", "InsectEPGSmallTrain": "sensor", "InsectWingbeatSound": "sensor",
    "ItalyPowerDemand": "device", "LargeKitchenAppliances": "device", "Lightning2": "sensor", "Lightning7": "sensor", "Mallat": "simulated",
    "Meat": "spectro", "MedicalImages": "image", "MiddlePhalanxOutlineAgeGroup": "image", "MiddlePhalanxOutlineCorrect": "image",
    "MiddlePhalanxTW": "image", "MixedShapesRegularTrain": "image", "MixedShapesSmallTrain": "image", "MoteStrain": "sensor",
    "NonInvasiveFetalECGThorax1": "sensor", "NonInvasiveFetalECGThorax2": "sensor", "OliveOil": "spectro", "OSULeaf": "image",
    "PhalangesOutlinesCorrect": "image", "Phoneme": "speech", "PickupGestureWiimoteZ": "motion", "PigAirwayPressure": "sensor",
    "PigArtPressure": "sensor", "PigCVP": "sensor", "PLAID": "device", "Plane": "simulated", "PowerCons": "device",
    "ProximalPhalanxOutlineAgeGroup": "image", "ProximalPhalanxOutlineCorrect": "image", "ProximalPhalanxTW": "image",
    "RefrigerationDevices": "device", "Rock": "spectro", "ScreenType": "device", "SemgHandGenderCh2": "sensor", "SemgHandMovementCh2": "sensor",
    "SemgHandSubjectCh2": "sensor", "ShakeGestureWiimoteZ": "motion", "ShapeletSim": "simulated", "ShapesAll": "image",
    "SmallKitchenAppliances": "device", "SmoothSubspace": "simulated", "SonyAIBORobotSurface1": "sensor", "SonyAIBORobotSurface2": "sensor",
    "StarLightCurves": "astronomy", "Strawberry": "image", "SwedishLeaf": "image", "Symbols": "image", "SyntheticControl": "simulated",
    "ToeSegmentation1": "image", "ToeSegmentation2": "image", "Trace": "sensor", "TwoLeadECG": "sensor", "TwoPatterns": "simulated",
    "UMD": "motion", "UWaveGestureLibraryAll": "motion", "UWaveGestureLibraryX": "motion", "UWaveGestureLibraryY": "motion",
    "UWaveGestureLibraryZ": "motion", "Wafer": "device", "Wine": "spectro", "WordSynonyms": "image", "Worms": "image",
    "WormsTwoClass": "image", "Yoga": "motion"
}