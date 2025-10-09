```markdown
# moneymaker — Automated Betting Analysis (API-Football)

Ez a projekt egy alapvető, kiterjeszthető fogadási elemző rendszer, amely az API-Football v3-at használja valós idejű mérkőzés- és statisztikai adatokhoz. A rendszer támogatja mind a Poisson alapú, mind a gépi tanulás alapú előrejelzéseket.

Fő cél:
- valós forma számítás (utolsó lezárt meccsek alapján)
- Poisson alapú lambda számítás és 1X2, BTTS, Over/Under becslés
- **ML alapú előrejelzések** LogisticRegression modellekkel (1X2, BTTS, Over/Under 2.5)
- Edge és Kelly alapú kockázatkezelési javaslat
- Automatikus fallback Poisson-ra ha ML modellek nem elérhetők
- Továbbfejleszthető kalibrációval, Bayesian modellekkel, és TippmixPro integrációval

Követelmények
- Python 3.10+
- API-Football kulcs (https://www.api-football.com/documentation-v3)
- Telepítés:
  ```bash
  pip install -r requirements.txt
  ```

Telepítés
1. Klónozd a repót:
   git clone https://github.com/kiss96dani/moneymaker.git
2. Hozz létre .env fájlt (lásd .env.example) és állítsd be az API_FOOTBALL_KEY értékét.
3. Telepítsd a függőségeket:
   pip install -r requirements.txt

Használat

## Alapvető elemzés (Poisson módszerrel)
- Mérkőzések letöltése és elemzése:
  ```bash
  python betting.py --fetch --analyze
  ```

- Csak elemzés meglévő adatokon:
  ```bash
  python betting.py --analyze
  ```

- Konkrét fixture-ök elemzése:
  ```bash
  python betting.py --analyze --fixture-ids 12345,67890
  ```

## Monte Carlo szimuláció
- Mérkőzések elemzése Monte Carlo Poisson-szimulációval (jobb megbízhatóság):
  ```bash
  python betting.py --fetch --analyze --use-mc
  ```

- Az iterációk számának módosítása (alapértelmezett: 10,000):
  ```bash
  python betting.py --fetch --analyze --use-mc --mc-iters 50000
  ```

A Monte Carlo módszer véletlenszerű Poisson-mintavételezéssel szimulál, és empirikus valószínűségeket számol az 1X2, BTTS és Over/Under 2.5 piacokra. A módszer előnye, hogy robusztusabb becslést ad, főleg szélsőséges lambda értékek esetén. Több iteráció pontosabb eredményeket ad, de lassabb futási idővel jár.

## ML modellek használata

### 1. Modellek betanítása
Első lépésként történeti adatokon kell betanítani a modelleket:

```bash
python betting.py --train-models --train-from 2023-01-01 --train-to 2023-12-31 --leagues 39,61
```

Paraméterek:
- `--train-models`: ML modellek betanítása
- `--train-from`: Kezdő dátum (YYYY-MM-DD)
- `--train-to`: Végző dátum (YYYY-MM-DD)
- `--leagues`: Liga ID-k vesszővel elválasztva (alapértelmezett: 39,61 - Premier League, Ligue 1)

**Fontos:** A training sok API hívást generál. Javasolt kisebb időszakkal kezdeni (pl. 6-12 hónap) és limitált liga számmal.

### 2. Elemzés ML modellekkel
Miután a modellek betanultak, elemzéskor használhatod őket:

```bash
python betting.py --fetch --analyze --use-ml
```

vagy

```bash
python betting.py --analyze --fixture-ids 12345,67890 --use-ml
```

Ha a modellek nem találhatók vagy hiba történik, automatikusan visszavált Poisson módszerre.

Kimenet
- data/fixtures/: raw fixture JSON fájlok
- data/analysis/: elemzési eredmények JSON fájlok
- daily_reports/: napi top mérkőzések összefoglalók (top_markets_YYYY-MM-DD.json)
- config/: konfigurációs adatok (pl. ligák listája)
- models/: betanított ML modellek (*.pkl fájlok)

## Új funkciók

### Head-to-Head (H2H) adatok
Az elemzések mostantól tartalmazzák az utolsó 10 H2H mérkőzés adatait a két csapat között. Ez az `h2h` mezőben található az analysis JSON-ben.

### Helyi időzóna (Budapest)
A mérkőzések kezdési ideje mostantól szerepel Budapest időzónában is (`kickoff_local` mező), az UTC mellett.

### Odds és Edge számítás
Az elemzések tartalmazzák a piaci oddsokat és a számított edge/Kelly értékeket minden elérhető piacon (1X2, BTTS, Over/Under 2.5). A top markets report minden tipp mellé csatolja:
- Piaci odds
- Implied probability
- Edge (model_prob × odds - 1)
- Kelly ajánlás (fractional Kelly és stake ajánlás)

Az edge számítás csak akkor történik, ha piaci odds elérhető az adott piachoz.

**Minimális odds szűrés**: A summarizer csak >= MIN_ODDS (alapértelmezett: 1.85) oddsokkal rendelkező tippeket listáz. Ez környezeti változóval módosítható:
```bash
export MIN_ODDS=2.0
python betting.py --analyze
```

## ML Pipeline technikai részletek

### Feature-ök
A ML modellek a következő feature-öket használják:
- `home_goals_per_match`: hazai csapat átlag góljai (utolsó 5 meccs)
- `home_goals_against_per_match`: hazai csapat kapott gólok átlaga
- `away_goals_per_match`: vendég csapat átlag góljai
- `away_goals_against_per_match`: vendég csapat kapott gólok átlaga
- `form_score_home`: súlyozott forma érték (W=1, D=0.5, L=0)
- `form_score_away`: vendég forma érték
- `home_adv`: hazai pálya előny konstans

### Modellek
- **1X2 model**: Multinomial LogisticRegression (3 osztály: home/draw/away)
- **BTTS model**: Binary LogisticRegression (2 osztály: yes/no)
- **Over/Under 2.5 model**: Binary LogisticRegression (2 osztály: over/under)

### Odds parsing és edge számítás
A rendszer kiterjesztett odds parsing-ot használ:
- 1X2 piac: home/draw/away odds
- BTTS piac: yes/no odds
- Over/Under 2.5 piac: over/under odds

Edge és Kelly stake minden piacon:
- Edge = model_prob × odds - 1
- Kelly = edge / (odds - 1) ha edge > 0

Megjegyzés
Ez a rendszer production-ready ML pipeline-t biztosít Poisson fallback-kel. Az "enhanced modeling" (kalibráció, Bayes, Monte Carlo) és a TippmixPro/Tippmix integráció továbbfejleszthető. A training sok API hívást generál - javasolt retry/backoff mechanizmus használata production környezetben.
```