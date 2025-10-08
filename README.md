```markdown
# moneymaker — Automated Betting Analysis (API-Football)

Ez a projekt egy alapvető, kiterjeszthető fogadási elemző rendszer, amely az API-Football v3-at használja valós idejű mérkőzés- és statisztikai adatokhoz, majd Poisson alapú modellekkel becsli a kimeneteleket.

Fő cél:
- valós forma számítás (utolsó lezárt meccsek alapján)
- Poisson alapú lambda számítás és 1X2, BTTS, Over/Under becslés
- Edge és Kelly alapú kockázatkezelési javaslat
- Később bővíthető kalibrációval, Bayesian modellekkel, és TippmixPro integrációval

Követelmények
- Python 3.10+
- API-Football kulcs (https://www.api-football.com/documentation-v3)
- Ajánlott:
  pip install -r requirements.txt

Telepítés
1. Klónozd a repót:
   git clone https://github.com/kiss96dani/moneymaker.git
2. Hozz létre .env fájlt (lásd .env.example) és állítsd be az API_FOOTBALL_KEY értékét.
3. Telepítsd a függőségeket:
   pip install -r requirements.txt

Használat
- Mérkőzések letöltése és elemzése:
  python betting.py --fetch --analyze

- Csak elemzés meglévő adatokon:
  python betting.py --analyze

- Konkrét fixture-ök elemzése:
  python betting.py --analyze --fixture-ids 12345,67890

Kimenet
- data/fixtures/: raw fixture JSON fájlok
- data/analysis/: elemzési eredmények JSON fájlok
- config/: konfigurációs adatok (pl. ligák listája)

Megjegyzés
Ez a kiinduló kód alapvető implementációt ad; az "enhanced modeling" (kalibráció, Bayes, Monte Carlo) és a TippmixPro/Tippmix integráció további implementációt igényel. A parserek a bookmakerek odds-ainak heterogén alakjai miatt egyszerűsítettek — a valós odds-források feldolgozásához kiterjesztés szükséges.
```