# Ausarbeitung — Prüfungsvorbereitung Prof. M.

**Machine Learning (192.183) · TU Wien · 2026S**
Teil I des Kurses: klassisches Machine Learning. Diese Ausarbeitung deckt genau die Themen ab, die M. prüft, mit Formeln zum Auswendiglernen, durchgerechneten Beispielen und den klassischen Prüfungsfallen.

---

## 0. Wie M. prüft (Strategie zuerst)

M.-Hälfte testet in einem anderen Stil als B.-Hälfte. Das ändert, wie man lernt.

- **Fragetypen:** True/False (mit Negativbewertung), kurze Handrechnungen, „erkläre / vergleiche". Du musst kleine Modelle **per Hand** bauen können (Decision Tree, kNN, Naive Bayes…).
- **Bewertung (wichtig!):** Bei True/False gilt typischerweise **+2 für richtig, 0 für keine Antwort, −1 für falsch**. Wenn du eine Frage wirklich nicht weißt, ist **leer lassen besser als raten**. Prüfungsanleitung lesen, um das Schema zu bestätigen.
- **Beste Vorbereitung:** Alte Prüfungen rechnen — die Fragen kommen fast wortgleich wieder (Baumtiefe-Berechnung, Micro- vs. Macro-Averaging, „was sind Landmarking-Features", z-score vs. min-max, Bandit-„welche Schritte waren exploratorisch").

**M. unterrichtet (laut Folien):** Lecture 1 Regression, Lecture 2 (1R, Naive Bayes, Covering), Lecture 5 (kNN, Decision Trees, Evaluation, Random Forests), Lecture 6 Reinforcement Learning (Bandits, MDPs, TD), Lecture 7 Metalearning & AutoML. Dazu klassisch: Ensembles, SVM, Preprocessing.

> **Achtung Querschneider:** *Unsupervised Learning (Clustering, PCA)* unterrichtet **B.**, nicht M. *Reinforcement Learning* ist geteilt: die **Grundlagen** (k-armed Bandit, ε-greedy, MDP, TD) sind M.; der Deep-Tail (Advantage, Actor-Critic, GRPO, RLVR) ist B.

---

## 1. Lineare & Polynomiale Regression

**Idee:** Regression beantwortet „wie viel?" — das Ziel ist eine kontinuierliche Zahl (Hauspreis, Laufzeit, CPU-Performance).

### Lineares Modell
$$\hat{y} = h_w(x) = w_0 + w_1 x_1 + \dots + w_n x_n = w^\top x$$
- $w_j$ = Gewicht von Feature $j$ (wie stark ändert sich $\hat y$ bei +1 Einheit in $x_j$)
- $w_0$ = Intercept/Bias (Vorhersage wenn alle Features 0). Trick: Dummy-Feature $x_0=1$.

### Kostenfunktion (RSS / MSE)
$$\text{RSS}(w) = \sum_{i=1}^m \big(y^{(i)} - h_w(x^{(i)})\big)^2$$
Residuen werden quadriert, damit positive/negative Fehler sich nicht aufheben und große Fehler hart bestraft werden. Perfekter Fit → Kosten 0.

### Gradientenabstieg
$$w_j \leftarrow w_j - \eta \frac{\partial}{\partial w_j} \text{Cost}(w)$$
$\eta$ = Lernrate. Iteratives Verfahren, wenn die geschlossene Lösung zu teuer ist.

### Normalengleichung (geschlossene Form)
$$w = (X^\top X)^{-1} X^\top y$$
Exakte Lösung in einem Schritt — aber $(X^\top X)^{-1}$ ist teuer bei vielen Features.

### Polynomiale Regression
Features werden um Potenzen erweitert ($x, x^2, x^3, \dots$). Das Modell bleibt **linear in den Parametern** $w$, ist aber nichtlinear in $x$. Höherer Grad → höhere Gefahr von Overfitting.

### Regularisierung
- **Ridge (L2):** $\text{RSS} + \lambda \sum w_j^2$ — schrumpft Gewichte, setzt sie aber nicht exakt auf 0.
- **Lasso (L1):** $\text{RSS} + \lambda \sum |w_j|$ — treibt manche Gewichte auf **exakt 0** → Feature-Selektion.

> **Prüfungsfalle:** F-Score ist eine **Klassifikations**metrik, **nicht** für Regression (→ Aussage „F-Score evaluiert Regression" ist FALSE). Regression nutzt MAE, MSE, RMSE.

---

## 2. Bayes'sches Lernen: Naive Bayes & Bayes-Netze

### Bayes-Regel
$$P(H \mid E) = \frac{P(E \mid H)\, P(H)}{P(E)}$$
Posterior = Likelihood × Prior / Evidence. $P(E)$ ist Normalisierungskonstante (gleich für alle Hypothesen → kürzt sich beim Vergleich weg).

### Naive Bayes
$$c^\star = \arg\max_c\ P(c) \prod_{k=1}^n P(E_k \mid c)$$
„Naiv" = Annahme, dass Features bei bekannter Klasse **bedingt unabhängig** sind. Nutzt — anders als 1R — **alle** Attribute. Naive Bayes ist ein **eager learner** (baut die Wahrscheinlichkeitstabellen beim Training).

**Durchgerechnet (Wetter-Datensatz, 9 yes / 5 no):**
Neuer Tag: Outlook=Sunny, Temp=Cool, Humidity=High, Windy=True.
$$L(\text{yes}) = \tfrac{2}{9}\cdot\tfrac{3}{9}\cdot\tfrac{3}{9}\cdot\tfrac{3}{9}\cdot\tfrac{9}{14} = 0{,}0053$$
$$L(\text{no}) = \tfrac{3}{5}\cdot\tfrac{1}{5}\cdot\tfrac{4}{5}\cdot\tfrac{3}{5}\cdot\tfrac{5}{14} = 0{,}0206$$
Normalisiert: $P(\text{no}) = 0{,}0206/(0{,}0053+0{,}0206) = 0{,}795$ → **Vorhersage: no**.

### Zero-Frequency-Problem & Laplace
Kommt ein Attributwert nie mit einer Klasse vor → $P=0$ → annulliert das ganze Produkt. Lösung — Add-One (Laplace):
$$P(E_k=v \mid c) = \frac{n_{c,v}+1}{n_c+V}$$
$V$ = Anzahl möglicher Werte des Attributs.
**Beispiel:** Overcast nie bei „no" → roh $0/5=0$. Mit Laplace: $P(\text{Overcast}\mid\text{no}) = (0+1)/(5+3) = 1/8 = 0{,}125$.

### Fehlende & numerische Werte
- **Fehlend:** den Faktor einfach weglassen (symmetrisch für beide Klassen → fair).
- **Numerisch:** Gauß annehmen, $\mu,\sigma$ pro Klasse schätzen, Dichte einsetzen:
$$f(x) = \frac{1}{\sqrt{2\pi}\,\sigma}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

### Bayes-Netze
DAG mit bedingten Wahrscheinlichkeitstabellen (CPTs). Naive Bayes ist der **Spezialfall**: ein Klassenknoten als Elternteil aller Feature-Knoten, keine Kanten zwischen Features.

**Drei Verbindungstypen & d-Separation:**
- **Seriell** $A \to B \to C$: abhängig; **unabhängig gegeben B** (mittlerer Knoten blockiert die Kette).
- **Divergierend** $A \leftarrow B \to C$: unabhängig gegeben B.
- **Collider** $A \to B \leftarrow C$: unabhängig **ohne** Bedingung; **abhängig gegeben B** (Beobachten des Colliders öffnet den Pfad!).

**Strukturlernen:** Hill-Climbing — leerer Graph, dann wiederholt Kante hinzufügen/löschen/umkehren (azyklisch halten), Zug mit bester BIC-Verbesserung behalten, bei lokalem Optimum stoppen. $\text{BIC} = \log\text{-Likelihood} - \tfrac{k}{2}\log m$.

> **Prüfungsfalle:** Ein allgemeines Bayes-Netz kann Naive Bayes **schlagen**, weil es Abhängigkeiten zwischen Features modellieren kann. NB ist zur bedingten Unabhängigkeit gezwungen.

---

## 3. Regellernen: 1R & Covering (PRISM)

### 1R (One-Rule)
Baut für **jedes** Attribut eine Ein-Attribut-Regel, wählt das Attribut mit der **niedrigsten Fehlerrate**. Holte (1993): erstaunlich oft fast so gut wie komplexe Modelle. **„Simplicity first."**
- 1R ist ein **Decision Tree der Tiefe 1** (ein Split, ein Blatt pro Wert).
- **Numerisch:** Werte sortieren, Breakpoints setzen wo die Mehrheitsklasse wechselt; Mindestanzahl pro Intervall (z.B. 3) gegen Overfitting.

### PRISM (Covering / Separate-and-Conquer)
Eine Klasse wählen, Regel wachsen lassen, die möglichst viele ihrer Beispiele abdeckt und andere ausschließt; abgedeckte Beispiele entfernen; wiederholen.
$$\text{Bedingung wählen, die } \frac{p}{t} \text{ maximiert}$$
- $t$ = abgedeckte Instanzen gesamt, $p$ = davon positive. Bedingungen hinzufügen bis $p/t=1$ (perfekt). Bei Gleichstand: größeres $t$ bevorzugen.

> **Vergleich Covering vs. Decision Trees:** Covering = separate-and-conquer (eine Klasse nach der anderen, geordnete IF-THEN-Liste). Trees = divide-and-conquer (alle Klassen gleichzeitig, sich gegenseitig ausschließende Pfade).

---

## 4. Decision Trees

Greedy, top-down (ID3/C4.5/CART): am besten bewerteten Split nehmen, rekursiv weiter, stoppen bei reinem Knoten oder Stoppregel. Jedes Blatt sagt die Mehrheitsklasse voraus. Geometrisch: achsen-parallele Splits → rechteckige Regionen.

### Split-Kriterien
**1. Misclassification Error:** $\text{Err} = 1 - \max_i p_i$. Einfach, aber schwacher Splitter (unempfindlich gegen Änderungen, die die Mehrheit nicht bewegen).

**2. Entropie & Information Gain:**
$$H(X) = -\sum_{i=1}^C p_i \log_2 p_i \qquad IG = H(X) - \sum_j \frac{|X_j|}{|X|} H(X_j)$$
Max-Entropie für $C$ gleichwahrscheinliche Klassen = $\log_2 C$ (2 Klassen → 1 bit). Wir wählen den Split mit dem **größten IG**.

**3. Gini-Impurity:**
$$I_G = 1 - \sum_{i=1}^C p_i^2$$
Max für 2 Klassen = 0,5. Kein Logarithmus → billiger. CART-Default.

**Durchgerechnetes Root-Beispiel (14 Zeilen, 9 yes / 5 no):**
$$H(\text{root}) = -\tfrac{9}{14}\log_2\tfrac{9}{14} - \tfrac{5}{14}\log_2\tfrac{5}{14} = 0{,}940 \text{ bits}$$
Attribut Price (low 4/0, medium 3/2, high 2/3): gewichtete Kinder-Entropie $= \tfrac{4}{14}(0) + \tfrac{5}{14}(0{,}971) + \tfrac{5}{14}(0{,}971) = 0{,}694$. → $IG(\text{Price}) = 0{,}940 - 0{,}694 = 0{,}246$ bits. Für jedes Attribut so rechnen, höchstes IG an die Wurzel.

### Gain Ratio (C4.5)
$$\text{GainRatio} = \frac{IG}{\text{SplitInfo}},\quad \text{SplitInfo} = -\sum_i \frac{|T_i|}{|T|}\log_2\frac{|T_i|}{|T|}$$
Korrigiert die Verzerrung von IG zugunsten von Attributen mit **vielen Werten** (ein ID-Attribut macht jedes Kind rein → IG maximal, aber nutzlos).

### Max-Depth-Berechnung (Klassiker!)
*Gegeben:* 1000 Beobachtungen; Split nur wenn Knoten ≥ 200 (min-to-split); jedes Blatt ≥ 300 (min-leaf). Maximale Tiefe ohne Wurzel?
**Logik:** Um zu splitten braucht ein Knoten **zwei** Kinder ≥ 300, also ≥ 600 (die Blatt-Regel dominiert). Möglichst schief splitten:
- Tiefe 0: $1000 \ge 600$ → 300 | 700
- Tiefe 1: $700 \ge 600$ → 300 | 400
- Tiefe 2: $400 < 600$ → Stopp.
**→ Maximale Tiefe = 2.** Immer prüfen, welche der beiden Schwellen bindet.

> **Prüfungsfallen:**
> - Baumtiefe kann **nie** $N-1$ überschreiten (jeder Split trennt ≥ 1 Beispiel). „Tiefer als Anzahl Samples" → FALSE.
> - Jeder Decision Tree lässt sich in eine **äquivalente Menge von IF-THEN-Regeln** umschreiben (ein Pfad = eine Regel) → TRUE.
> - Verschiedene Kriterien (Error/Entropie/Gini) können **verschiedene Bäume** liefern.

**Stärken/Schwächen:** interpretierbar, kein Scaling nötig, schnell ↔ überfittet leicht, instabil (kleine Datenänderung → anderer Baum), nur achsen-parallel, greedy.

---

## 5. k-Nearest Neighbours (kNN)

**Idee:** Neuen Punkt durch Mehrheit seiner $k$ nächsten Trainingsnachbarn klassifizieren. **Lazy learner** (keine Trainingsarbeit, alles zur Query-Zeit) — Gegenteil zu eager (Tree, Perceptron).

### Distanzfunktionen
$$\text{Euklid: } \sqrt{\textstyle\sum_i (a_i-b_i)^2}\quad \text{Manhattan: } \textstyle\sum_i |a_i-b_i|\quad \text{Cosine: } 1 - \frac{\sum a_i b_i}{\|x\|\|y\|}$$

### Effekt von k
Kleines $k$ → flexibel, rauschempfindlich (overfit). Großes $k$ → glatter, kann unterfit­ten. Bei Gleichstand kann sich die Vorhersage zwischen $k=1$ und $k=3$ umkehren.

**kNN für Regression:** Vorhersage = Mittelwert (oder gewichteter Mittelwert) der $k$ Nachbarwerte.

> **Prüfungsfalle:** **Feature-Scaling ist entscheidend.** Ungeskalierte Features mit großem Wertebereich dominieren die Distanz. Beschleunigung: k-d-Tree.

---

## 6. Modell-Evaluation & statistische Tests

**Warum Test-Set:** Misst man auf Trainingsdaten, belohnt man Memorisieren statt Generalisieren. Split: Train (fittet Parameter) / Validation (tunt Hyperparameter, Early Stopping) / Test (einmal am Ende).

### Confusion Matrix
| | Predicted + | Predicted − |
|---|---|---|
| **Actual +** | TP | FN |
| **Actual −** | FP | TN |

- **FP = Type-I-Error** (Fehlalarm), **FN = Type-II-Error** (verpasst). *Klassischer Verwechslungsfehler!*

### Metriken
$$\text{Accuracy} = \frac{TP+TN}{TP+TN+FP+FN}\quad \text{Precision} = \frac{TP}{TP+FP}\quad \text{Recall} = \frac{TP}{TP+FN}$$
$$F_1 = \frac{2PR}{P+R} = \frac{2TP}{2TP+FP+FN}$$

**Durchgerechnet** ($TP=40, FP=10, FN=20, TN=30$): Accuracy 0,70; Precision 0,80; Recall 0,667; $F_1 = 0{,}727$.

> Accuracy lügt bei unbalancierten Klassen (99 % „gesund" → 99 % Accuracy, nutzlos). Precision/Recall decken das auf.

### Micro- vs. Macro-Averaging (Multi-Class)
- **Macro:** Metrik pro Klasse, dann ungewichtet mitteln → jede Klasse zählt gleich (gut für seltene Klassen).
- **Micro:** alle TP/FP/FN poolen, dann einmal rechnen → jedes Beispiel zählt gleich, häufige Klassen dominieren. (Single-Label: Micro-Precision = Micro-Recall = Accuracy.)

**Durchgerechnet** (A: 90/10, B: 1/9, C: 2/8):
$$P_{\text{macro}} = \tfrac{0{,}90+0{,}10+0{,}20}{3} = 0{,}40,\quad P_{\text{micro}} = \tfrac{93}{120} = 0{,}775$$
Die große Lücke ist der Kern der Frage: Micro vom leichten A dominiert, Macro deckt schwaches B/C auf.

### Cross-Validation
k-fold: in $k$ Folds teilen, abwechselnd auf $k-1$ trainieren / 1 testen, Scores mitteln. Jedes Beispiel genau einmal getestet → geringere Varianz. **LOOCV** = Extremfall $k=n$ (fast unbiased, aber teuer, hohe Varianz).

### Regression-Metriken
$$\text{MAE} = \tfrac{1}{n}\sum|y_i-\hat y_i|\quad \text{MSE} = \tfrac{1}{n}\sum(y_i-\hat y_i)^2\quad \text{RMSE} = \sqrt{\text{MSE}}$$

> **Prüfungsfalle:** MSE/RMSE (quadrieren) sind **empfindlicher gegen Outlier** als MAE. „MAE empfindlicher als RMSE" → FALSE.

### Signifikanztests
- **Paired t-Test:** zwei Modelle über dieselben CV-Folds.
- **McNemar-Test:** zwei Klassifizierer auf einem Test-Set (diskordante Counts $b,c$). Statistik $\frac{(|b-c|-1)^2}{b+c}$ gegen kritischen Wert 3,84 (α=0,05, 1 df).

### Overfitting
Niedriger Trainings-, hoher Testfehler (wachsende Lücke). Gegenmittel: mehr Daten, Regularisierung (L1/L2), Pruning, einfacheres Modell, CV + Early Stopping.

---

## 7. Ensemble-Methoden

**Warum:** Bias-Varianz-Tradeoff. Diversität reduziert die Varianz des Ensembles.

### Bagging (Bootstrap Aggregating)
$T$ Modelle auf je einer Bootstrap-Stichprobe; Regression → Mittelwert, Klassifikation → Mehrheit. Jeder Baum verpasst ≈ 37 % der Daten → **Out-Of-Bag (OOB)** als gratis Validierung.

### Random Forests
Bagging + ein Extra-Trick.
> **Prüfungsfalle — „Was ist die Zufälligkeit?"** Zwei Quellen (beide nennen!):
> 1. **Bootstrap-Sampling** der Daten (Bagging-Teil).
> 2. **Zufällige Feature-Teilmenge** an jedem Split — nur $m$ von $p$ Features als Kandidaten.

Default: $m=\lfloor\sqrt p\rfloor$ (Klassifikation), $m=\lfloor p/3\rfloor$ (Regression). Kleineres $m$ → mehr Dekorrelation. $m=p$ → reduziert sich auf Bagging.

### Boosting
Sequentiell: jedes Modell korrigiert die Fehler des vorigen. **AdaBoost** (Gewichte auf falsch klassifizierte Beispiele erhöhen) vs. **Gradient Boosting** (auf Residuen fitten). Gewichtete Stimmen — Gegensatz zu Bagging (gleiche Stimmen).

---

## 8. Lineare Klassifizierer: Perceptron, SVM, Kernel

### Perceptron
Lineare Trenngrenze $w^\top x + b = 0$. Update bei Fehlklassifikation: $w \leftarrow w + \eta\, y_i x_i$. Konvergiert nur bei linear separierbaren Daten.

### SVM
Maximiert den **Margin** (Abstand der Trenngrenze zu den nächsten Punkten = Support Vectors). **Soft Margin** ($C$-Parameter, Slack-Variablen) für nicht-separable Daten.

### Kernel
„Kernel-Trick" — nichtlineare Grenzen ohne explizite Transformation, durch Ersetzen des Skalarprodukts (z.B. RBF, polynomial). Kann auch auf das Perceptron angewandt werden.

> **Perceptron vs. SVM:** Beide linear; SVM wählt **eindeutig** die margin-maximale Grenze, Perceptron irgendeine trennende.

---

## 9. Feature Selection & Preprocessing

### Scaling: z-score vs. min-max
$$\text{z-score: } x' = \frac{x-\mu}{\sigma}\ (\text{Mittel 0, Std 1, unbeschränkt})\qquad \text{min-max: } x' = \frac{x-\min}{\max-\min}\ (\in[0,1])$$
z-score bei ~gaußschen Daten / Distanz- & Gradientenmethoden / Outliern; min-max wenn beschränkter Bereich gebraucht.

> **Prüfungsfalle:** Skalierungsparameter ($\mu,\sigma$ bzw. min,max) **nur am Trainingsset** berechnen — sonst Data Leakage. Min-max ist **outlier-empfindlicher** als z-score → TRUE.

### Kategorische Features
**One-Hot** statt Integer-Encoding für nominale Features (Integer erfindet falsche Ordnung & Abstände).

### Fehlende Werte
Löschen oder Imputation (Mean/Median numerisch, Modus kategorisch). **Median robuster gegen Outlier** als Mean.

### Feature Selection — drei Familien
- **Filter** (z.B. Information Gain, Korrelation) — schnell, modell-agnostisch.
- **Wrapper** (Forward Selection, RFE) — trainiert Modell auf Teilmengen, genau aber teuer.
- **Embedded** (Lasso L1, Tree-Importances) — Selektion während des Trainings.

### Information Gain vs. PCA (Klassiker)
$$IG(Y,X) = H(Y) - H(Y\mid X)$$
> **Prüfungsfalle:** „PCA ist unsupervised, Information Gain ist supervised" → TRUE. IG nutzt die Labels $y$ (misst Reduktion der Label-Entropie, wählt Original-Features). PCA ignoriert $y$ völlig (maximiert Varianz in $X$, **erzeugt neue** Komponenten = Feature-**Extraktion**, nicht -Selektion). PCA kann ein niedrig-varianten aber hoch-prädiktiven Feature verwerfen.

**Data Augmentation:** label-erhaltende Transformationen (Rotation, Flip, Crop) — nur auf **Trainingsdaten**.

---

## 10. Metalearning, AutoML & Hyperparameter-Optimierung

### No Free Lunch (NFL)
$$\sum_{\text{alle } f} \text{Perf}(A_1,f) = \sum_{\text{alle } f} \text{Perf}(A_2,f)$$
Über **alle** möglichen Probleme gemittelt sind alle Algorithmen gleich gut. → Kein universell bester Lerner; man muss pro Datensatz auswählen.
> **Prüfungsfalle:** NFL sagt **nicht**, dass alle Algorithmen auf *deinem* Problem gleich gut sind — reale Probleme sind eine strukturierte Teilmenge.

### Rice's Framework (Algorithmenauswahl)
Vier Räume + Selektionsabbildung: **Problem space P** (Datensätze) → **Feature space F** (Meta-Features $f(x)$) → **Algorithm space A** → **Performance space Y**.
$$\alpha = S(f(x)) = \arg\max_{a\in A} \widehat{\text{Perf}}(a, f(x))$$
Beim Metalearning ist $S$ selbst ein gelerntes Modell.

### Meta-Features — vier Familien
1. **Statistisch/informationstheoretisch** (Anzahl Attribute/Klassen, Klassen-Entropie, Korrelation…).
2. **Modell-basiert** (Tiefe/Knotenzahl eines auf den Daten gebauten Baums).
3. **Landmarking** (s.u.).
4. **Learning Curves** (Form der Accuracy-vs-Trainingsgröße-Kurve).

> **Landmarking-Features (wiederkehrende Frage):** die **Accuracies schneller, einfacher Lerner** (Naive Bayes, 1-NN, OneR, Decision Stump), als Deskriptoren genutzt. Aussagekräftig, weil der Erfolg eines simplen Lerners die Geometrie des Problems verrät (hohe NB-Accuracy → Klassen fast separierbar; hohe OneR → ein Feature genügt).

### CASH & AutoML
**Combined Algorithm Selection and Hyperparameter optimisation:** Algorithmus-Wahl ist eine **kategorische Top-Level-Hyperparameter** in einem bedingten Suchraum. Algorithmenwahl und HPO sind **vereint**, nicht getrennt.

### HPO
$$\lambda^\star = \arg\min_{\lambda\in\Lambda} f(\lambda),\quad f(\lambda) = \tfrac{1}{k}\sum_i \text{Loss}(A_\lambda \text{ auf Fold } i)$$
- **Grid Search:** alle Kombinationen — exponentielle Kosten.
- **Random Search:** zufällig — schlägt Grid oft, weil es mehr **distinkte Werte der wenigen wichtigen** Hyperparameter probiert.
- **Bayesian Optimisation (SMBO):** Surrogat-Modell von $f(\lambda)$ + Acquisition-Funktion (Expected Improvement). SMAC nutzt Random-Forest-Surrogat. Lohnt nur wenn jede Auswertung teuer ist.

---

## 11. Reinforcement Learning — M.-Teil (Grundlagen)

**RL** = weder Labels noch fester Datensatz, nur ein Reward-Signal; der Agent erzeugt seine Daten durch Handeln. Kern: **Exploration vs. Exploitation**.

**Vier Elemente:** Policy $\pi(a\mid s)$ (was tun), Reward $r$ (was jetzt gut ist), Value $v(s)$ (erwarteter Langzeit-Reward = Weitsicht), Model (optional, Umweltdynamik).

### k-Armed Bandit
Ein Zustand, $k$ Arme, unbekannte mittlere Rewards.
$$q_*(a) = \mathbb{E}[R\mid A=a],\qquad Q_t(a) \approx q_*(a)$$

**ε-greedy:** mit Wahrscheinlichkeit $1-\varepsilon$ greedy ($\arg\max_a Q_t(a)$), mit $\varepsilon$ zufälliger Arm.

**Inkrementelles Update (Sample-Average):**
$$Q_{n+1} = Q_n + \tfrac{1}{n}\big(R_n - Q_n\big)$$
$[R_n - Q_n]$ = Fehler/Überraschung. Schrittweite $1/n$ = gleicher Reward-Gewicht; konstantes $\alpha$ für nicht-stationäre Probleme (jüngere Rewards zählen mehr).

**Smartere Exploration:** Optimistic Initial Values (alle $Q_0$ hoch → jeder Arm enttäuscht → systematisch erkunden); **UCB:** $A_t = \arg\max_a [Q_t(a) + c\sqrt{\ln t / N_t(a)}]$.

**Durchgerechneter Prüfungstyp — „welche Schritte waren definitiv exploratorisch?":**
Bei sample-average ε-greedy, $Q_1(a)=0$. Man trackt $Q_t(a)$ für jeden Arm und prüft an jedem Schritt: war die gewählte Aktion **nicht** die mit dem höchsten $Q$? Dann muss es der ε-Fall (Zufall) gewesen sein. Methodik: Q-Tabelle Schritt für Schritt updaten, an jedem $t$ den greedy-Arm bestimmen, mit der tatsächlich gewählten Aktion vergleichen. Weicht sie ab → **definitiv exploratorisch**.

### Markov Decision Process (MDP)
Agent-Umwelt-Schleife: beobachtet $S_t$, wählt $A_t$, erhält $R_{t+1}, S_{t+1}$.

**Discounted Return:**
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots = \sum_{k=0}^\infty \gamma^k R_{t+k+1}$$
$\gamma\in[0,1]$ = Discount (nähere Rewards zählen mehr; $\gamma<1$ hält die Summe endlich).

**Value-/Q-Funktion:** $v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]$, $q_\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t=s, A_t=a]$.

**Temporal-Difference (TD):**
$$V(S_t) \leftarrow V(S_t) + \alpha\big[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD-Target}} - V(S_t)\big]$$
Lernt aus jedem Schritt (kein Warten auf das Episodenende, anders als Monte Carlo).

---

## 12. Schnelle True/False-Checkliste (häufige Fallen)

- F-Score evaluiert Regression → **FALSE** (Klassifikation).
- Baum tiefer als Anzahl Trainingsbeispiele → **FALSE** (≤ N−1).
- Decision Tree ↔ IF-THEN-Regelmenge → **TRUE**.
- MAE empfindlicher gegen Outlier als RMSE → **FALSE**.
- Min-max outlier-empfindlicher als z-score → **TRUE**.
- PCA unsupervised, Information Gain supervised → **TRUE**.
- Naive Bayes ist lazy → **FALSE** (eager).
- Type-I = FP, Type-II = FN.
- Random-Forest-Zufall = Bootstrap **+** Feature-Subset (beide!).
- NFL → alle Algorithmen auf meinem Problem gleich gut → **FALSE**.
- Collider $A\to B\leftarrow C$: gegeben B werden A, C **abhängig** (nicht unabhängig).

---

## 13. Empfohlene Lernreihenfolge

1. Diese Ausarbeitung einmal komplett durchlesen (Intuition + Formeln).
2. Handrechnungen mit Stift und Papier nachvollziehen: Naive Bayes (Wetter), IG/Entropie-Split, Max-Depth, Confusion-Matrix-Metriken, Micro/Macro, Bandit-Q-Update.
3. **3–4 alte M.-Prüfungen** rechnen — die Fragen wiederholen sich fast wortgleich.
4. True/False-Checkliste oben als letzte Wiederholung vor der Prüfung.
5. Negativbewertung beachten: bei echtem Nichtwissen leer lassen.

---

# Teil B — Altprüfungen ausgearbeitet (Fragen + Antworten)

> Dieser Teil sammelt die rekonstruierten Fragen aus mehreren Altprüfungen, geordnet nach Prüfungsstruktur. **TRUE/FALSE-Antworten** sind fett markiert, mit kurzer Begründung. Einige Fragen betreffen B.-Deep-Learning-Hälfte (CNN, Vanishing Gradients, Dropout) — sie tauchen aber faktisch in derselben Prüfung auf, daher sind sie hier mit beantwortet und als *(B.-Teil)* gekennzeichnet.

## B.1 Prüfungsstruktur (typisch)

- **Teil 1 (26 P):** 13 × True/False. **+2** richtig, **−1** falsch, **0** leer.
- **Teil 2 (12 P):** 4 × Handrechnung (kein Taschenrechner). **+3** richtig, **−1,5** falsch, **0** leer.
- **Teil 3 (12 P):** Multiple Choice, „alle richtigen ankreuzen". **+2 nur wenn ALLE** korrekten Optionen markiert sind, sonst **0**.
- Gesamtzeit ~75 min, drei Gruppen.

---

## B.2 Teil 1 — True/False (mit Begründung)

**Grundlagen / Klassifikation**

1. *Classification ist eine ML-Aufgabe, bei der das Zielattribut nominal ist.* → **TRUE.** Klassifikation sagt eine nominale (kategorische) Klasse voraus; Regression sagt eine numerische Größe voraus.
2. *Decision Trees können nur binäre Klassifikationsprobleme lösen.* → **FALSE.** Bäume handhaben Multi-Class problemlos (ein Blatt pro Klasse möglich).
3. *Der Fehler eines 1-NN-Klassifizierers auf dem Trainingsset ist 0.* → **TRUE.** Der nächste Nachbar eines Trainingspunkts ist er selbst (Distanz 0) → immer korrekt klassifiziert.
4. *Die Softmax-Funktion in MLPs transformiert die Aktivierung in den Bereich −1…1.* → **FALSE.** Softmax liefert Werte in **(0,1)**, die sich zu 1 summieren (Wahrscheinlichkeiten). −1…1 ist tanh.
5. *Ordinaldaten erlauben keine Distanzberechnung zwischen Datenpunkten.* → **TRUE.** Ordinal hat eine Ordnung (Rang), aber keine sinnvollen numerischen Abstände.

**Evaluation**

6. *Macro-Averaging berechnet zuerst Accuracy/Precision/Recall pro Klasse und mittelt dann über die Klassen.* → **TRUE.** Genau das ist die Definition (jede Klasse zählt gleich).
7. *Der Paired t-Test wird bei Holdout-Validierung verwendet.* → **FALSE.** Der Paired t-Test vergleicht zwei Modelle über dieselben **Cross-Validation-Folds**.
8. *Der Paired t-Test wird bei Cross-Validation verwendet.* → **TRUE.** (siehe oben). Für ein einzelnes Test-Set nimmt man McNemar.
9. *In einem Datensatz ist die Entropie am niedrigsten, wenn alle Klassen gleich viele Samples haben.* → **FALSE.** Niedrigste Entropie (0) bei **reinem** Knoten (eine Klasse). Gleichverteilung = maximale Entropie.
10. *In einem Datensatz ist die Entropie am höchsten, wenn alle Klassen gleich viele Samples haben.* → **TRUE.** Max-Entropie $\log_2 C$ bei Gleichverteilung.

**Ensembles & SVM**

11. *In AdaBoost werden die Gewichte zufällig initialisiert.* → **FALSE.** Sie werden **uniform** initialisiert ($1/N$ für jedes Beispiel).
12. *Das erste Modell in Gradient Boosting ist ein Zero-Rule-Modell.* → **TRUE.** Gradient Boosting startet mit einer konstanten Vorhersage (Mittelwert bei Regression / Log-Odds), also einem ZeroR-Modell, und fittet dann auf die Residuen.
13. *SVMs finden immer eine optimalere Trenngrenze (Hyperplane) als Perceptrons.* → **FALSE.** „Immer optimaler" ist zu stark. Auf separierbaren Daten findet das Perceptron *eine* gültige Trenngrenze; SVM wählt die **margin-maximale** (nach diesem Kriterium besser), aber das ist nicht generell „optimaler" für jedes Ziel.
14. *SVMs mit linearem Kernel eignen sich besonders für sehr hochdimensionale, sparse Daten.* → **TRUE.** Klassischer Fall: Textklassifikation (Bag-of-Words) ist hochdimensional/sparse — linearer SVM ist dort sehr stark.
15. *SVMs können standardmäßig nur binäre Klassifikation lösen.* → **TRUE.** SVM ist von Haus aus binär; Multi-Class via one-vs-rest / one-vs-one.

**Naive Bayes / Bayes-Netze / Metalearning**

16. *Wird Naive Bayes auf einen Datensatz mit numerischen Attributen angewandt, muss immer eine Wahrscheinlichkeitsdichtefunktion verwendet werden.* → **FALSE.** Man kann numerische Attribute auch **diskretisieren** (in Intervalle binnen) statt eine Gauß-Dichte anzunehmen.
17. *Modell-basierte Features für Metalearning werden direkt aus dem Datensatz extrahiert.* → **FALSE.** Sie werden aus einem **auf dem Datensatz trainierten Modell** abgeleitet (z.B. Tiefe/Knotenzahl eines Baums), nicht direkt aus den Rohdaten.
18. *Ein Bayes-Netz ist ein gerichteter zyklischer Graph.* → **FALSE.** Es ist ein gerichteter **azyklischer** Graph (DAG).
19. *Ein Decision Tree kann in eine Regelmenge umgewandelt werden.* → **TRUE.** Ein Pfad Wurzel→Blatt = eine IF-THEN-Regel.

**Reinforcement Learning**

20. *Bei der Monte-Carlo-Methode im RL werden Value-Schätzungen und Policy nur bei Abschluss einer Episode geändert.* → **TRUE.** MC braucht den vollständigen Return einer Episode; Updates erfolgen erst am Episodenende (anders als TD, das schrittweise updatet).
21. *k-armed Bandits wählen die nächste Aktion basierend auf dem erwarteten zukünftigen Reward.* → **FALSE (Vorsicht/Nuance).** Bandits haben nur einen Zustand und keinen „zukünftigen" Return — sie wählen nach dem geschätzten **erwarteten (sofortigen) Reward** $Q_t(a)$. „Zukünftiger Reward" ist MDP-Sprache (discounted return), nicht Bandit.

**Regression / Feature Selection**

22. *Gradient Descent ist für lineare Regression immer effizienter als die Normalengleichung (analytisch).* → **FALSE.** Bei wenigen Features ist die Normalengleichung (geschlossene Lösung) effizienter; Gradient Descent gewinnt erst bei sehr vielen Features (wo $(X^\top X)^{-1}$ teuer wird).
23. *Information Gain ist eine unsupervised Feature-Selection-Methode.* → **FALSE.** IG nutzt die Labels ($H(Y)-H(Y|X)$) → **supervised**.
24. *PCA ist eine supervised Feature-Selection-Methode.* → **FALSE.** PCA ist **unsupervised** und ist Feature-**Extraktion** (neue Komponenten), nicht -Selektion.
25. *Feature Selection dient primär dazu, die Effektivität (Genauigkeit) von ML zu verbessern.* → **FALSE (Nuance).** Primär verbessert sie **Effizienz** und Interpretierbarkeit und bekämpft Overfitting; Genauigkeitsgewinn ist ein möglicher Nebeneffekt, nicht das Primärziel.
26. *Majority Voting wird nicht verwendet, wenn kNN für Regression eingesetzt wird.* → **TRUE.** kNN-Regression **mittelt** die Nachbarwerte; Mehrheitsvotum gibt es nur bei Klassifikation.

---

## B.3 Teil 2 — Handrechnungen (Methode + durchgerechnet)

### B.3.1 Naive Bayes: neue Zeile klassifizieren (ohne Laplace)

**Datensatz (7 Trainingsinstanzen), klassifiziere Instanz 8:**

| Inst | F1 | F2 | F3 | Klasse |
|---|---|---|---|---|
| 1 | a | y | k | − |
| 2 | c | y | s | − |
| 3 | a | y | k | + |
| 4 | b | n | k | − |
| 5 | b | y | s | + |
| 6 | a | n | s | + |
| 7 | b | n | s | − |
| **8** | **a** | **n** | **s** | **?** |

Klassenzählung: **−** = {1,2,4,7} (4), **+** = {3,5,6} (3). Priors: $P(-)=4/7$, $P(+)=3/7$.

**Klasse −** (F1=a, F2=n, F3=s):
$P(a\mid-)=1/4$ (nur Inst 1), $P(n\mid-)=2/4$ (Inst 4,7), $P(s\mid-)=2/4$ (Inst 2,7).
$$L(-) = \tfrac{4}{7}\cdot\tfrac{1}{4}\cdot\tfrac{2}{4}\cdot\tfrac{2}{4} = \tfrac{4}{7}\cdot\tfrac{1}{16} = \tfrac{1}{28} \approx 0{,}0357$$

**Klasse +**:
$P(a\mid+)=2/3$ (Inst 3,6), $P(n\mid+)=1/3$ (Inst 6), $P(s\mid+)=2/3$ (Inst 5,6).
$$L(+) = \tfrac{3}{7}\cdot\tfrac{2}{3}\cdot\tfrac{1}{3}\cdot\tfrac{2}{3} = \tfrac{3}{7}\cdot\tfrac{4}{27} = \tfrac{4}{63} \approx 0{,}0635$$

$L(+) > L(-)$ → **Vorhersage: +**.

### B.3.2 Decision Tree mit Entropie: neue Zeile klassifizieren

Gleicher Datensatz. **Methode:** Root-Attribut über IG wählen, dann ggf. weiter splitten, neue Instanz den Pfad hinunter schicken.

Root-Entropie: $H = -\tfrac{4}{7}\log_2\tfrac{4}{7} - \tfrac{3}{7}\log_2\tfrac{3}{7} \approx 0{,}985$ bits.

IG für F1 (a: 1−,3+,6+ → 1/3 rein-ish; b: 4−,5+,7− ; c: 2−):
- a (3 Inst, 2+/1−): $H=0{,}918$; b (3 Inst, 1+/2−): $H=0{,}918$; c (1 Inst, rein): $H=0$.
- gewichtet $= \tfrac{3}{7}(0{,}918)+\tfrac{3}{7}(0{,}918)+\tfrac{1}{7}(0) = 0{,}787$ → $IG(F1)=0{,}198$.

IG für F3 (k: 1−,3+,4− → 1+/2−, $H=0{,}918$; s: 2−,5+,6+,7− → 2+/2−, $H=1$):
- gewichtet $=\tfrac{3}{7}(0{,}918)+\tfrac{4}{7}(1)=0{,}965$ → $IG(F3)=0{,}020$.

F1 hat den höchsten IG → **Wurzel = F1**. Neue Instanz hat **F1=a**; im a-Zweig sind {1:−, 3:+, 6:+} → Mehrheit **+** (weiter auf F2/F3 splitten bestätigt dies, da n→+ und s→+). → **Vorhersage: +**.

> Merke: Tritt im Zweig ein Gleichstand auf, weiter auf das nächstbeste Attribut splitten; sonst Mehrheitsklasse des Blatts.

### B.3.3 k-armed Bandit: welche Schritte waren zufällig (exploratorisch)?

$k=5$, $Q_1(a)=0$ für alle $a$, ε-greedy, **Sample-Average**. Sequenz:
$A_1{=}1,R_1{=}{-}1$; $A_2{=}2,R_2{=}{-}1$; $A_3{=}2,R_3{=}2$; $A_4{=}1,R_4{=}{-}1$; $A_5{=}2,R_5{=}4$; $A_6{=}5,R_6{=}3$.

**Methode:** vor jedem Schritt den greedy-Arm ($\arg\max Q$) bestimmen; weicht die gewählte Aktion davon ab → **definitiv exploratorisch**. (Ist sie greedy-konsistent, *kann* sie greedy gewesen sein → nicht definitiv zufällig.)

| t | Q vorher [1,2,3,4,5] | greedy-Arm(e) | gewählt | Verdikt |
|---|---|---|---|---|
| 1 | [0,0,0,0,0] | alle (tie) | 1 | greedy möglich |
| 2 | [−1,0,0,0,0] | 2,3,4,5 | 2 | greedy möglich |
| 3 | [−1,−1,0,0,0] | 3,4,5 | 2 | **zufällig** |
| 4 | [−1,0.5,0,0,0] | 2 | 1 | **zufällig** |
| 5 | [−1,0.5,0,0,0] | 2 | 2 | greedy möglich |
| 6 | [−1,1.67,0,0,0] | 2 | 5 | **zufällig** |

(Updates: nach t2 hat Arm2 $-1$; nach t3 Arm2 $\{-1,2\}\to 0{,}5$; nach t5 Arm2 $\{-1,2,4\}\to 1{,}67$.)

**→ Definitiv zufällig: t3, t4, t6.**

### B.3.4 Regression: MAE / RMSE berechnen

**Formeln:** $\text{MAE}=\tfrac{1}{n}\sum|y_i-\hat y_i|$, $\text{RMSE}=\sqrt{\tfrac{1}{n}\sum(y_i-\hat y_i)^2}$.

**Beispiel-Methode** für eine Funktion $\hat y = 3 + 2\,F_1 + F_2$ mit 4 Instanzen:

| F1 | F2 | $\hat y = 3+2F_1+F_2$ | $y$ (wahr) | $\|y-\hat y\|$ |
|---|---|---|---|---|
| 1 | 0 | 5 | 4 | 1 |
| 2 | 1 | 8 | 10 | 2 |
| 0 | 2 | 5 | 8 | 3 |
| 1 | 1 | 6 | 8 | 2 |

$$\text{MAE} = \frac{1+2+3+2}{4} = \frac{8}{4} = 2$$

> Vorgehen in der Prüfung: (1) für jede Zeile $\hat y$ aus der Funktion einsetzen, (2) absolute Residuen bilden, (3) mitteln. Für RMSE die Residuen **quadrieren**, mitteln, Wurzel ziehen. Antwortoptionen sind meist {2,5 / 2 / 1,5 / none} — exakt rechnen, nicht schätzen.

### B.3.5 1R: welches Feature wählt 1R?

**Methode:** Für jedes Attribut die beste Ein-Attribut-Regel bilden (pro Wert Mehrheitsklasse), Fehler summieren. Attribut mit **kleinstem Fehler** gewinnt. Mit obigem 7-Instanzen-Datensatz:

- **F1:** a→{1−,3+,6+} pred + (1 Fehler); b→{4−,5+,7−} pred − (1 Fehler); c→{2−} pred − (0). **Fehler = 2/7.**
- **F2:** y→{1−,2−,3+,5+} tie (2 Fehler); n→{4−,6+,7−} pred − (1). **Fehler = 3/7.**
- **F3:** k→{1−,3+,4−} pred − (1); s→{2−,5+,6+,7−} tie (2). **Fehler = 3/7.**

**→ 1R wählt F1** (niedrigster Fehler 2/7).

### B.3.6 *(B.-Teil)* Convolution-Output-Größe & Max-Pooling

**Formel (pro Dimension):** $O = \left\lfloor \dfrac{W - K + 2P}{S}\right\rfloor + 1$.

- Beispiel Input 32, $K=5, P=0, S=1$ → $O=28$.
- $K=5, P=0, S=2$ → $O=\lfloor 13{,}5\rfloor+1 = 14$.
- „Same" $K=5, P=2, S=1$ → $O=32$.
- **Max-Pool** $2\times2$, $S=2$: halbiert → $28\to14$. Pooling hat **0 trainierbare Gewichte** und ändert die Kanalzahl nicht.

---

## B.4 Teil 3 — Multiple Choice (alle richtigen ankreuzen)

### B.4.1 Welche sind bekannte CNN-Architekturen?
**Richtig: LeNet, ResNet** (und Inception, falls „Reception" ein Tippfehler dafür ist).
Falsch: **LSTM** (rekurrentes Netz, kein CNN), **TeNet** (Fantasiename). *(B.-Teil)*

### B.4.2 Der Output eines Convolutional Layers ist größer, wenn …
Aus $O=\frac{W-K+2P}{S}+1$:
**Richtig: Padding nimmt zu** (größeres $P$ → größeres $O$) **und Stride nimmt ab** (kleineres $S$ → größeres $O$).
Falsch: Padding nimmt ab; Stride nimmt zu. *(B.-Teil)*

### B.4.3 Welche Methoden helfen gegen Vanishing Gradients?
Toolbox (alle ankreuzen, die vorkommen): **gute Initialisierung** ($1/\sqrt d$, He), **Residual/Skip-Connections**, **Normalisierung** (BatchNorm/LayerNorm), **nicht-saturierende Aktivierungen** (ReLU), **LSTM-Gating** (bei Sequenzen). **Gradient Clipping** wird oft mitgelistet — es zielt eigentlich auf *exploding* gradients, steht aber in derselben Toolbox. *(B.-Teil)*

### B.4.4 Welche Klassifikationsmethoden nutzen Majority Voting?
**Richtig: k-NN** (Mehrheit der $k$ Nachbarn) und **Random Forests** (Mehrheit der Bäume).
Falsch: einzelner Decision Tree (kein Voting), Bayesian Network, und das Ensemble, das die Vorhersage des Modells mit der **höchsten Confidence** ausgibt (das ist Max-Confidence-Selektion, kein Majority Vote). „All of the above" und „None" daher falsch.

### B.4.5 Welche Methoden verhindern Overfitting?
**Richtig: Regularisierung** (L1/L2), **Dropout**, **Cross-Validation** (zur Modell-/Hyperparameterwahl & Erkennung), zusätzlich oft **Data Augmentation** und **Early Stopping**. **Batch Normalization** hat einen leichten regularisierenden Nebeneffekt, dient aber primär der Trainingsstabilität. *(teils B.-Teil)*

### B.4.6 Wofür kann ein Validation-Set verwendet werden?
**Richtig: Early Stopping** und **Hyperparameter-Tuning**.
Falsch: **Signifikanztests** (dafür braucht man Test-Set/CV-Folds).

### B.4.7 Welche Technik ähnelt Dropout in neuronalen Netzen?
**Richtig: Bagging.** Beide trainieren viele Modelle auf perturbierten Versionen und mitteln; Dropouts Modelle sind gewichtsteilende Subnetze. *(B.-Teil)*

### B.4.8 Umgang mit fehlenden Werten in Trainings- und Test-Set
**Richtige Vorgehensweise:** löschen oder **imputieren** (Mean/Median numerisch, Modus kategorisch). Wichtig: die **Imputationsstatistik nur am Trainingsset** berechnen und dieselben Werte auf das Test-Set anwenden (sonst Data Leakage). Median ist robuster gegen Outlier als Mean.

---

## B.5 Offene Fragen — Musterantworten

**Können Kernel-Methoden auch auf das Perceptron angewandt werden? (mit Begründung)**
Ja. Das Perceptron lässt sich in „dualer" Form schreiben, in der nur **Skalarprodukte** zwischen Datenpunkten vorkommen. Ersetzt man das Skalarprodukt durch eine Kernel-Funktion $K(x,x')$, erhält man ein **Kernel-Perceptron**, das nichtlineare Trenngrenzen lernen kann — derselbe Trick wie bei der SVM.

**Was ist Polynomiale Regression? Vor-/Nachteile gegenüber linearer Regression?**
Lineare Regression mit um Potenzen ($x^2, x^3, \dots$) erweiterten Features; bleibt linear in den Parametern. **Vorteil:** kann nichtlineare Zusammenhänge modellieren (Kurven). **Nachteile:** höhere Gefahr von **Overfitting** bei hohem Grad, schlechtere Extrapolation, schwerer interpretierbar.

**Wann sind zwei Knoten in einem Bayes-Netz d-separiert?**
Wenn **jeder** Pfad zwischen ihnen „blockiert" ist: bei seriellen/divergierenden Verbindungen blockiert ein **beobachteter** Zwischenknoten; bei einem Collider blockiert ein **unbeobachteter** Collider (dessen Nachkommen ebenfalls unbeobachtet). D-separiert ⇒ bedingt unabhängig.

**Welche Features im Metalearning? Was sind Landmarking-Features?**
Vier Familien: statistisch/informationstheoretisch, modell-basiert, **Landmarking**, Learning Curves. Landmarking-Features sind die **Accuracies schneller, einfacher Lerner** (Naive Bayes, 1-NN, OneR, Decision Stump) auf dem Datensatz — sie charakterisieren die Geometrie des Problems (z.B. hohe OneR-Accuracy ⇒ ein Feature genügt).

**Unterschied Micro- vs. Macro-Averaging?**
Macro: Metrik pro Klasse, dann ungewichtet mitteln (jede Klasse gleich → seltene Klassen sichtbar). Micro: alle TP/FP/FN poolen, dann einmal rechnen (jedes Beispiel gleich → häufige Klassen dominieren; = Accuracy bei Single-Label).

**Max-Depth mit Pre-Pruning (1000 Obs, min-split 200, min-leaf 300)?**
Zum Splitten braucht ein Knoten zwei Kinder ≥ 300, also **≥ 600**. 1000→(300|700)→(300|400); 400 < 600 → Stopp. **Max-Tiefe = 2** (ohne Wurzel). Die Blatt-Regel (≥300) bindet, nicht die schwächere Split-Regel (≥200).

**Local-Search-Algorithmus für Bayes-Netz-Erstellung?**
Hill-Climbing: mit leerem Graph starten; wiederholt eine Kante **hinzufügen/löschen/umkehren** (azyklisch halten); den Zug behalten, der einen Score wie **BIC** ($\log$-Likelihood − Komplexitätsstrafe) am meisten verbessert; bei lokalem Optimum stoppen. Random Restarts helfen gegen schlechte Optima.

**Ziel & Setting von Klassifikation — Abgrenzung zu anderen ML-Aufgaben?**
Ziel: aus gelabelten Trainingsdaten eine Funktion lernen, die neuen Instanzen eine **nominale Klasse** zuordnet (supervised). Verwandt mit **Regression** (auch supervised, aber numerisches Ziel). Unterscheidet sich von **Clustering** (unsupervised, keine Labels) und **Reinforcement Learning** (nur Reward-Signal, keine Labels).

**Methoden gegen Overfitting in neuronalen Netzen?**
Mehr Daten / Data Augmentation, **Dropout**, L1/L2-Regularisierung (Weight Decay), **Early Stopping**, **Batch/Layer-Normalization**, kleineres Modell. *(B.-Teil)*

**Drei Methoden zur Fehlerberechnung in Regression?**
**MAE** $=\tfrac1n\sum|y-\hat y|$ (robust gegen Outlier), **MSE** $=\tfrac1n\sum(y-\hat y)^2$ (straft große Fehler stark), **RMSE** $=\sqrt{\text{MSE}}$ (gleiche Einheit wie $y$). (RSS = unnormierte Summe.)

**z-score vs. min-max — was, wann, für welche Features?**
z-score $x'=(x-\mu)/\sigma$ → Mittel 0, Std 1, unbeschränkt; gut bei ~gaußschen Daten, Distanz-/Gradientenmethoden und Outliern. min-max $x'=(x-\min)/(\max-\min)$ → Bereich [0,1]; gut bei gewünschtem beschränktem Bereich, milden Outliern. Beide nur für **numerische** Features; Parameter nur am Trainingsset schätzen.

**ε-greedy beim k-armed Bandit erklären.**
Mit Wahrscheinlichkeit $1-\varepsilon$ wird die Aktion mit dem höchsten geschätzten Wert gewählt (**Exploitation**), mit Wahrscheinlichkeit $\varepsilon$ ein zufälliger Arm (**Exploration**). $\varepsilon$ steuert den Trade-off; $\varepsilon=0$ = rein greedy (riskiert, einen anfangs unglücklichen, aber guten Arm nie wieder zu testen).

**Was ist Data Augmentation?**
Erzeugen neuer Trainingsbeispiele durch **label-erhaltende Transformationen** (Rotation, Flip, Crop, Helligkeits-Jitter), v.a. bei Bildern — vergrößert die Datenmenge und macht das Modell robuster. Nur auf **Trainingsdaten** anwenden. *(B.-Teil)*

---

*Viel Erfolg bei der Prüfung!*
