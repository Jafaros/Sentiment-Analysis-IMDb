# Závěr

V této práci jsem porovnal dvě metody sentimentální analýzy, konkrétně VADER a TF-IDF + Logistic Regression. Níže jsou uvedené výsledky a krátké zhodnocení.

## Výsledky

Grafy a podrobnější výsledky se nachází ve složce `outputs`.

### VADER

| Měřený údaj | Skóre  |
| ----------- | ------ |
| Accuracy    | 0.6977 |
| Precision   | 0.6506 |
| Recall      | 0.8543 |
| F1 skóre    | 0.7386 |

$$
\begin{bmatrix}
6764 & 5736 \\
1821 & 10679
\end{bmatrix}
$$

<p style="text-align: center;">Confusion Matrix pro VADER</p>

### TF-IDF + Logistic Regression

| Měřený údaj | Skóre  |
| ----------- | ------ |
| Accuracy    | 0.8967 |
| Precision   | 0.8931 |
| Recall      | 0.9013 |
| F1 skóre    | 0.8972 |

$$
\begin{bmatrix}
11152 & 1348 \\
1234 & 11266
\end{bmatrix}
$$

<p style="text-align: center;">Confusion Matrix pro TF-IDF</p>

## Otázky

### Která metoda vyšla lépe a proč?

Lépe vyšla metoda TF-IDF + Logistic Regression. Je vidět, že má lepší výsledky skoro ve všem, hlavně v accuracy a F1 skóre. Podle mě je to tím, že se model učí přímo z trénovacích recenzí, takže si lépe "zapamatuje", jaká slova a slovní spojení bývají u pozitivních a negativních hodnocení. VADER je oproti tomu spíš obecný nástroj a jede hlavně podle slovníku, takže tolik nevnímá celý kontext věty.

### Jaké typické chyby dělá VADER vs TF-IDF?

VADER často chyboval u delších recenzí, kde se míchaly pozitivní i negativní výrazy. Často se nechal zmást tím, že se v negativní recenzi objevila nějaká kladně znějící slova, a pak ji vyhodnotil špatně. Problém mu podle mě dělá i ironie nebo situace, kdy recenze nejdřív něco pochválí, ale celkově film stejně zhodnotí negativně.

TF-IDF si vedl lépe, ale ani on nebyl bez chyb. Nejvíc se mýlil u recenzí, které byly napsané trochu zvláštně, byly hodně popisné nebo měly jasné hodnocení až úplně na konci. Někdy ho taky zmátla slova, která sama o sobě zní pozitivně nebo negativně, ale v celém kontextu znamenají něco jiného.

### Uveďte 2 příklady (pos+neg) chybně klasifikovaných recenzí u každé metody.

#### VADER

1. Pozitivní recenze chybně klasifikovaná jako negativní:

Soubor `test/pos/10006_7.txt` byl ve skutečnosti pozitivní, ale VADER ho vyhodnotil jako negativní. V recenzi se sice objevuje kritika jako "really cheesy sometimes", ale celkově autor film spíš doporučuje, například větou "if it's on the television, check it out".

2. Negativní recenze chybně klasifikovaná jako pozitivní:

Soubor `test/neg/10000_4.txt` byl ve skutečnosti negativní, ale VADER ho vyhodnotil jako pozitivní. Recenze přitom jasně říká třeba "Generic and boring" nebo "second-rate action trash", jenže v textu jsou i slova, která zní pozitivně, a to VADER zřejmě zmátlo.

#### TF-IDF + Logistic Regression

1. Pozitivní recenze chybně klasifikovaná jako negativní:

Soubor `test/pos/10018_8.txt` byl ve skutečnosti pozitivní, ale model ho označil jako negativní. V textu se objevují slova jako "bloody" nebo "nightmarish", která působí negativně, i když celkový závěr recenze je kladný: "I like this one a lot, check it out. 8 out of 10."

2. Negativní recenze chybně klasifikovaná jako pozitivní:

Soubor `test/neg/10003_3.txt` byl ve skutečnosti negativní, ale model ho označil jako pozitivní. Nejspíš ho zmátly výrazy jako "alluring visual qualities" nebo "attractive", i když celkově je recenze záporná a obsahuje i věty jako "they fail miserably".

### Jak by se dal VADER optimalizovat pro tento účel?

VADER by se podle mě dal zlepšit hlavně tím, že by se upravila hranice pro rozhodování podle `compound` skóre, protože nula nemusí být pro tento dataset nejlepší. Taky by pomohlo přidat do slovníku víc slov typických pro filmové recenze, třeba "cheesy", "overrated" nebo "masterpiece". Další věc je, že VADER umí pracovat i s interpunkcí a zvýrazněním, takže by možná nebylo ideální mu to při předzpracování všechno mazat. A nakonec by šlo doplnit jednoduchá pravidla třeba pro negaci nebo pro poslední větu v recenzi, protože právě tam bývá často finální názor autora.
