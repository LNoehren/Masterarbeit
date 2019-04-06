# Vocalfolds SemSeg Masterarbeit

Code zur Masterarbeit **Entwicklung und Vergleich verschiedener deep learning Algorithmen 
zur semantischen Segmentierung eines Medizinischen Datensatzes** von Lennard Nöhren

Um ein Training durchzuführen muss die _main.py_ Datei ausgeführt werden. Dabei muss als 
Argument eine oder mehrere config yaml Datei angegeben werden. In dem Ordner _configs_ 
sind einige Beispiel-Configdateien bereitgestellt.

In der _Configuration_ Klasse ist im Detail beschrieben welche Felder in den config Dateien
erlaubt sind.

Unter anderem muss die Netzwerkarchitektur, die verwendet werden soll angegeben werden. In 
dem _models_ Ordner sind die Implementierungen der verschiedenen Architekturen enthalten. 
Es stehen U-Net, SegNet, E-Net, ERFNet und DeepLabV3+ zur Verfügung. Bei DeepLab kann 
außerdem das backbone Model ausgetauscht werden. Dafür stehen implementierungen von dem 
originalen ResNet101, Xception und eine veränderte Version von ResNet101 in der 
_deeplab_v3_plus_ Datei zu Verfügung.

Gibt man in der config mehrere Netzwerkarchitekturen an wird eine Ensemble Struktur aus
ihnen aufgebaut. Dabei werden die Parameter der einzelnen Netzwerke eingefroren und ein
Gewichtsvektor, der trainiert werden muss wird hinzugefügt. Die ausgabe des Ensembles ist
dann die gewichtete Summe der einzelnen Netzwerke.

Zum Training der Netzwerke gibt es eine abstrakte _Model_ Klasse, bei der die genutzte
Architektur ausgetauscht werden kann. Es wird immer der Adam optimizer genutzt. Außerdem 
wird eine gewichtete categorical cross entropy als Loss Funktion und Intersection over 
Union als Metrik verwendet.

für das Einlesen der Trainingsdaten gibt es eine _data_generator_ Klasse. Diese liest die
Bild- und Ground Truth Daten ein und kann auch Augmentierungen während dem Training auf
den Daten durchführen. Um die Performance zu verbessern wird dabei multiprocessing 
eingesetzt.

Gibt man in der config einen _load_path_ an können Gewichte aus einem vorigen Training
geladen werden. Dabei kann man auch angeben, ob das Training ein Pre-Training auf dem
Cityscapes oder dem ImageNet Datensatz war. Pre-Training mit ImageNet ist allerdings nur
für DeepLabV3+ unterstützt.

Die Fortschritte des Trainings werden in einem Ergebnisordner unter 
`results/<architektur>_<datum>_<startzeit>/` gespeichert. Es werden eine csv Datei, die 
den Loss und die Metrik enthält, die config Datei die genutzt wurde, einige Histogramme, 
Checkpoints von den Modellparametern und Daten für tensorboard gespeichert.

Weitere Dateien:
+ _utils.py_: Enthällt verschiedene Hilfsfunktionen, z.B. Funktionen zum Einlesen von 
Bildern.
+ _augmentations.py_: Enthällt Funktionen die zur Datenaugmentierung genutzt werden können.
+ _layers.py_: Enthällt verschiedene Layer und Blöcke, die in den Netzwerken genutzt werden.

Außerdem gibt es noch einen Ordner der verschiedene Skripte enthällt. Unter anderem sind
dort Skripte zum Testen der Inferenz, oder zum Augmentiern von Datensätzen zu finden.

Für das Training auf ImageNet wurde auch ein eigenes Skript entiwckelt, da es dort einige
Unterschiede zu einem Training für semantische Segmentierung gibt. Die dafür genutzten
Dateien sind unter _scripts/ImageNet_ zu finden.