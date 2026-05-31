"""
UCR Dataset Information Module

This module contains the list of all UCR time series classification datasets
and their domain classifications.
"""

all_ucr_datasets = [
    "Adiac", "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car", "CBF", "ChlorineConcentration", "CinCECGTorso", "Coffee",
    "Computers", "CricketX", "CricketY", "CricketZ", "DiatomSizeReduction", "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW", "Earthquakes", "ECG200",
    "ECG5000", "ECGFiveDays", "ElectricDevices", "FaceAll", "FaceFour", "FacesUCR", "FiftyWords", "Fish", "FordA", "FordB",
    "GunPoint", "Ham", "HandOutlines", "Haptics", "Herring", "InlineSkate", "InsectWingbeatSound", "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2",
    "Lightning7", "Mallat", "Meat", "MedicalImages", "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect", "MiddlePhalanxTW", "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme", "Plane", "ProximalPhalanxOutlineAgeGroup", "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices", "ScreenType",
    "ShapeletSim", "ShapesAll", "SmallKitchenAppliances", "SonyAIBORobotSurface1", "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf", "Symbols", "SyntheticControl",
    "ToeSegmentation1", "ToeSegmentation2", "Trace", "TwoLeadECG", "TwoPatterns", "UWaveGestureLibraryAll", "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ", "Wafer",
    "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga", "ACSF1", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ", "BME",
    "Chinatown", "Crop", "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "EOGHorizontalSignal", "EOGVerticalSignal", "EthanolLevel", "FreezerRegularTrain", "FreezerSmallTrain",
    "Fungi", "GestureMidAirD1", "GestureMidAirD2", "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung", "HouseTwenty",
    "InsectEPGRegularTrain", "InsectEPGSmallTrain", "MelbournePedestrian", "MixedShapesRegularTrain", "MixedShapesSmallTrain", "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", "PigCVP", "PLAID",
    "PowerCons", "Rock", "SemgHandGenderCh2", "SemgHandMovementCh2", "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "SmoothSubspace", "UMD"
]

dataset_domains = {
    # Sensor data
    "Adiac": "image", "ArrowHead": "image", "Beef": "spectro", "BeetleFly": "image", "BirdChicken": "image", "Car": "sensor", "CBF": "simulated", "ChlorineConcentration": "sensor", "CinCECGTorso": "sensor", "Coffee": "spectro",
    "Computers": "device", "CricketX": "motion", "CricketY": "motion", "CricketZ": "motion", "DiatomSizeReduction": "image", "DistalPhalanxOutlineAgeGroup": "image", "DistalPhalanxOutlineCorrect": "image", "DistalPhalanxTW": "image", "Earthquakes": "sensor", "ECG200": "ecg",
    "ECG5000": "ecg", "ECGFiveDays": "ecg", "ElectricDevices": "device", "FaceAll": "image", "FaceFour": "image", "FacesUCR": "image", "FiftyWords": "image", "Fish": "image", "FordA": "sensor", "FordB": "sensor",
    "GunPoint": "motion", "Ham": "spectro", "HandOutlines": "image", "Haptics": "motion", "Herring": "image", "InlineSkate": "motion", "InsectWingbeatSound": "sensor", "ItalyPowerDemand": "sensor", "LargeKitchenAppliances": "device", "Lightning2": "sensor",
    "Lightning7": "sensor", "Mallat": "simulated", "Meat": "spectro", "MedicalImages": "image", "MiddlePhalanxOutlineAgeGroup": "image", "MiddlePhalanxOutlineCorrect": "image", "MiddlePhalanxTW": "image", "MoteStrain": "sensor", "NonInvasiveFetalECGThorax1": "ecg", "NonInvasiveFetalECGThorax2": "ecg",
    "OliveOil": "spectro", "OSULeaf": "image", "PhalangesOutlinesCorrect": "image", "Phoneme": "sensor", "Plane": "sensor", "ProximalPhalanxOutlineAgeGroup": "image", "ProximalPhalanxOutlineCorrect": "image", "ProximalPhalanxTW": "image", "RefrigerationDevices": "device", "ScreenType": "device",
    "ShapeletSim": "simulated", "ShapesAll": "image", "SmallKitchenAppliances": "device", "SonyAIBORobotSurface1": "sensor", "SonyAIBORobotSurface2": "sensor", "StarLightCurves": "sensor", "Strawberry": "spectro", "SwedishLeaf": "image", "Symbols": "image", "SyntheticControl": "simulated",
    "ToeSegmentation1": "motion", "ToeSegmentation2": "motion", "Trace": "sensor", "TwoLeadECG": "ecg", "TwoPatterns": "simulated", "UWaveGestureLibraryAll": "motion", "UWaveGestureLibraryX": "motion", "UWaveGestureLibraryY": "motion", "UWaveGestureLibraryZ": "motion", "Wafer": "sensor",
    "Wine": "spectro", "WordSynonyms": "image", "Worms": "motion", "WormsTwoClass": "motion", "Yoga": "image", "ACSF1": "device", "AllGestureWiimoteX": "sensor", "AllGestureWiimoteY": "sensor", "AllGestureWiimoteZ": "sensor", "BME": "simulated",
    "Chinatown": "traffic", "Crop": "image", "DodgerLoopDay": "sensor", "DodgerLoopGame": "sensor", "DodgerLoopWeekend": "sensor", "EOGHorizontalSignal": "eog", "EOGVerticalSignal": "eog", "EthanolLevel": "spectro", "FreezerRegularTrain": "sensor", "FreezerSmallTrain": "sensor",
    "Fungi": "hrm", "GestureMidAirD1": "trajectory", "GestureMidAirD2": "trajectory", "GestureMidAirD3": "trajectory", "GesturePebbleZ1": "sensor", "GesturePebbleZ2": "sensor", "GunPointAgeSpan": "motion", "GunPointMaleVersusFemale": "motion", "GunPointOldVersusYoung": "motion", "HouseTwenty": "device",
    "InsectEPGRegularTrain": "epg", "InsectEPGSmallTrain": "epg", "MelbournePedestrian": "traffic", "MixedShapesRegularTrain": "image", "MixedShapesSmallTrain": "image", "PickupGestureWiimoteZ": "sensor", "PigAirwayPressure": "hemodynamics", "PigArtPressure": "hemodynamics", "PigCVP": "hemodynamics", "PLAID": "device",
    "PowerCons": "power", "Rock": "spectrum", "SemgHandGenderCh2": "spectrum", "SemgHandMovementCh2": "spectrum", "SemgHandSubjectCh2": "spectrum", "ShakeGestureWiimoteZ": "sensor", "SmoothSubspace": "simulated", "UMD": "simulated"
}