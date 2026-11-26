# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "decd1f9b-af0a-4944-b4cb-4e3e54161283",
# META       "default_lakehouse_name": "CommunityEvents",
# META       "default_lakehouse_workspace_id": "b91cc733-64db-452e-a99d-b82bc6214795",
# META       "known_lakehouses": [
# META         {
# META           "id": "decd1f9b-af0a-4944-b4cb-4e3e54161283"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # Let's Build Our First Fabric App: demo setup (Community Events)
# **AI @Work · Let's Build Something** — builds everything Act 1 starts from: a lakehouse with fictional event data, and a published semantic model that Power BI Desktop for Apps connects to.
# 
# **Before you run this notebook**
# 1. Use a workspace on **Fabric capacity** (a trial works for rehearsal). This is the same workspace your apps will deploy to.
# 2. Create a lakehouse named **`CommunityEvents`** and attach it to this notebook as the default lakehouse.
# 3. Select **Run all**. It takes about five minutes.
# 
# **What it creates** (all data is fictional)
# 
# | Item | What it holds |
# |---|---|
# | `sessions` table | 56 sessions in 6 tracks, in English, Spanish and Portuguese, with capacity |
# | `speakers` table | 40 fictional speakers and their community role |
# | `registrations` table | About 19,700 registrations, with attendee region and whether they attended |
# | `feedback` table | About 2,000 session ratings, 1 to 5 |
# | `dates` table | A date table for the model |
# | `Community Events` semantic model | Direct Lake model with relationships and 9 measures, ready for Act 1 |
# 
# The data is seeded and uses fixed dates (a fictional fall 2026 event season), so **every presenter gets the same numbers**. The answer key in section 5 is what a correct app should show.
# 
# This uses semantic link and the open-source semantic-link-labs library. Run it once in your dry run: library versions change, and you want to know before you're live.


# CELL ********************

%pip install -q semantic-link-labs

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 1. Settings
# Keep these names exactly as written. The runbook, the prompts and the reset step all use them.

# CELL ********************

LAKEHOUSE = "CommunityEvents"
MODEL = "Community Events"
TABLES = ["sessions", "speakers", "registrations", "feedback", "dates"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 2. Generate the data
# Nothing to change here.

# CELL ********************

# Community Events demo data generator (shared by the notebook and local tests)
import numpy as np, pandas as pd, datetime as dt

SEED = 2026
SEASON_START = dt.date(2026, 10, 5)
SEASON_END = dt.date(2026, 12, 11)

def build():
    rng = np.random.default_rng(SEED)

    # ---- speakers ----
    first = ["Ana","Bruno","Carla","Diego","Elena","Felipe","Grace","Hiro","Isabel","Jonas","Keiko","Lucas","Maya","Nadia","Omar","Priya",
             "Quinn","Rafael","Sofia","Tomas","Uma","Victor","Wen","Ximena","Yusuf","Zoe","Aiden","Beatriz","Chen","Daniela","Emeka","Fatima",
             "Gabriel","Hannah","Ivan","Julia","Kofi","Laura","Mateo","Nora"]
    last = ["Silva","Garcia","Okafor","Tanaka","Novak","Rossi","Kim","Haddad","Costa","Muller","Patel","Lopez","Nguyen","Ibrahim","Santos",
            "Smith","Fernandes","Kowalski","Ali","Moreno","Chen","Dubois","Mensah","Ortiz","Sato","Andersen","Reyes","Khan","Pereira","Jansen",
            "Torres","Yilmaz","Martins","Schmidt","Rivera","Lindqvist","Abe","Ramos","Nakamura","Vega"]
    ns = 40
    speakers = pd.DataFrame(dict(
        speaker_id=np.arange(1, ns+1),
        speaker_name=[f"{first[i]} {last[(i*7) % len(last)]}" for i in range(ns)],
        community_role=rng.choice(["MVP","Super User","Microsoft","Community member"], ns, p=[0.40,0.20,0.25,0.15]),
        speaker_region=rng.choice(["North America","Latin America","Europe","Asia Pacific"], ns, p=[0.35,0.25,0.25,0.15])))

    # ---- sessions ----
    tracks = {"Fabric Apps":["Apps for Report Builders","My First App in an Hour","Forms, Votes and Trackers","App Sharing Basics","From Prompt to Production"],
              "Power BI":["Measures Made Simple","Charts People Remember","Modeling Clinic","Copilot Tips for Report Authors","Page Layout Makeover","Paginated Reports Today"],
              "AI and Data":["Grounding AI in Your Data","Clean Data, Smart Answers","Agents for Analysts","Business Terms for AI"],
              "Data Engineering":["Lakehouse Fundamentals","Notebooks for BI Folks","Pipelines That Don't Break","Medallion in Practice","Spark for Beginners"],
              "Real-Time":["Live Dashboards","Streaming Hands-On","Alerts That Act"],
              "Community":["Contest Tips from Judges","Your First Talk","Starting a User Group","Ask the Super Users"]}
    langs = ["English","Spanish","Portuguese"]
    rows=[]; sid=1
    days=[SEASON_START+dt.timedelta(days=int(d)) for d in range((SEASON_END-SEASON_START).days+1)]
    weekdays=[d for d in days if d.weekday()<4]
    for track,titles in tracks.items():
        for t in titles:
            for lang in langs:
                if lang!="English" and rng.random()<0.45: continue
                cap = int(rng.choice([300,500,750,1000], p=[0.3,0.35,0.25,0.10]))
                rows.append(dict(session_id=sid, session_title=t, track=track, language=lang,
                    session_format=rng.choice(["Build It","Show It","Learn It"], p=[0.4,0.25,0.35]),
                    session_date=weekdays[int(rng.integers(0,len(weekdays)))],
                    speaker_id=int(rng.integers(1,ns+1)), capacity=cap))
                sid+=1
    sessions=pd.DataFrame(rows).sort_values(["session_date","session_id"]).reset_index(drop=True)
    sessions["session_id"]=np.arange(1,len(sessions)+1)

    # popularity: Fabric Apps and Fabric IQ are hot; non-English sessions smaller but loyal
    pop = sessions.track.map({"Fabric Apps":1.15,"AI and Data":1.0,"Power BI":0.85,"Data Engineering":0.6,"Real-Time":0.5,"Community":0.55}).values
    pop = pop * np.where(sessions.language=="English",1.0,0.7) * rng.uniform(0.6,1.25,len(sessions))
    n_reg = np.minimum((sessions.capacity.values*pop).astype(int), (sessions.capacity.values*1.0).astype(int))
    n_reg = np.maximum(n_reg, 40)

    # ---- registrations ----
    region_by_lang = {"English":(["North America","Europe","Asia Pacific","Latin America","Middle East & Africa"],[0.45,0.25,0.15,0.08,0.07]),
                      "Spanish":(["Latin America","Europe","North America"],[0.75,0.15,0.10]),
                      "Portuguese":(["Latin America","Europe"],[0.85,0.15])}
    regs=[]
    for s,n in zip(sessions.itertuples(), n_reg):
        rgs,p = region_by_lang[s.language]
        lead = rng.integers(1,42,n)
        att_p = 0.38 + (0.08 if s.session_format=="Build It" else 0) + (0.06 if s.language!="English" else 0)
        regs.append(pd.DataFrame(dict(session_id=s.session_id,
            attendee_region=rng.choice(rgs,n,p=p),
            registered_date=[s.session_date-dt.timedelta(days=int(x)) for x in lead],
            attended=rng.random(n)<att_p)))
    registrations=pd.concat(regs,ignore_index=True)
    registrations.insert(0,"registration_id",np.arange(1,len(registrations)+1))

    # ---- feedback (from about 22% of attendees) ----
    att=registrations[registrations.attended]
    fb=att[rng.random(len(att))<0.22]
    base=sessions.set_index("session_id").loc[fb.session_id]
    mu = 4.35 + np.where(base.session_format.values=="Build It",0.15,0) + rng.normal(0,0.05,len(fb))
    weak = sessions.session_id.isin(sessions.sample(6, random_state=SEED).session_id)   # a few sessions that need help
    mu = mu - np.where(fb.session_id.isin(sessions.session_id[weak]).values, 0.9, 0)
    rating=np.clip(np.round(rng.normal(mu,0.7)),1,5).astype(int)
    feedback=pd.DataFrame(dict(feedback_id=np.arange(1,len(fb)+1), session_id=fb.session_id.values,
        rating=rating, would_recommend=rating>=4))

    dates=pd.DataFrame(dict(date=pd.date_range(SEASON_START-dt.timedelta(days=45),SEASON_END).date))
    dates["year"]=[d.year for d in dates.date]; dates["month"]=[d.strftime("%b") for d in dates.date]
    dates["month_number"]=[d.month for d in dates.date]
    dates["week_start"]=[d-dt.timedelta(days=d.weekday()) for d in dates.date]
    dates["weekday"]=[d.strftime("%a") for d in dates.date]
    return dict(sessions=sessions, speakers=speakers, registrations=registrations, feedback=feedback, dates=dates)

def answer_key(t):
    s,r,f=t["sessions"],t["registrations"],t["feedback"]
    out={}
    out["sessions"]=len(s); out["speakers"]=s.speaker_id.nunique()
    out["registrations"]=len(r); out["attendees"]=int(r.attended.sum())
    out["attendance_rate"]=r.attended.mean(); out["avg_rating"]=f.rating.mean(); out["responses"]=len(f)
    reg=r.groupby("session_id").size(); fill=(reg/s.set_index("session_id").capacity)
    out["sessions_over_90_fill"]=int((fill>=0.9).sum())
    rt=f.groupby("session_id").rating.agg(["mean","count"])
    out["sessions_under_4"]=int(((rt["mean"]<4)&(rt["count"]>=10)).sum())
    out["by_language"]=r.merge(s,on="session_id").groupby("language").agg(registrations=("registration_id","size"),attendance=("attended","mean"))
    out["by_track"]=r.merge(s,on="session_id").groupby("track").agg(registrations=("registration_id","size")).sort_values("registrations",ascending=False)
    tops=pd.DataFrame(dict(registrations=reg)).join(s.set_index("session_id")[["session_title","language","capacity"]]).sort_values("registrations",ascending=False)
    out["top_sessions"]=tops.head(5)
    low=rt[rt["count"]>=10].join(s.set_index("session_id")[["session_title","language"]]).sort_values("mean").head(6)
    out["low_rated"]=low
    return out


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

spark.sql("DROP TABLE IF EXISTS sessions")
spark.sql("DROP TABLE IF EXISTS speakers")
spark.sql("DROP TABLE IF EXISTS registrations")
spark.sql("DROP TABLE IF EXISTS feedback")
spark.sql("DROP TABLE IF EXISTS dates")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

tables = build()
for name, df in tables.items():
    print(f"{name:14s} {len(df):>7,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 3. Write the tables to the lakehouse

# CELL ********************

# Clear old logical folders in the Lakehouse
import notebookutils

for tbl in ["sessions", "speakers", "registrations", "feedback", "dates"]:
    path = f"Tables/{tbl}"
    if notebookutils.fs.exists(path):
        notebookutils.fs.rm(path, recurse=True)
        print("Removed path:", path)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F

date_cols = {"sessions": ["session_date"], "registrations": ["registered_date"], "dates": ["date", "week_start"]}
for name, pdf in tables.items():
    pdf = pdf.copy()
    for c in date_cols.get(name, []):
        pdf[c] = pd.to_datetime(pdf[c])
    sdf = spark.createDataFrame(pdf)
    for c in date_cols.get(name, []):
        sdf = sdf.withColumn(c, F.to_date(c))
    sdf.write.mode("overwrite").option("overwriteSchema", "true").format("delta").saveAsTable(name)
    print(f"Wrote {name}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 4. Create the `Community Events` semantic model
# A Direct Lake model on the lakehouse, with relationships and measures that have plain-language descriptions. Descriptions matter: Desktop for Apps and Copilot read them to understand what each number means.

# CELL ********************

import sempy_labs as labs
from sempy_labs import directlake
from sempy_labs.tom import connect_semantic_model

directlake.generate_direct_lake_semantic_model(
    dataset=MODEL,
    source=LAKEHOUSE,   
    tables=TABLES,
    overwrite=True,
    refresh=False,
)

# Measures
M = [
    ("Sessions", "COUNTROWS(sessions)", "#,0", "Number of sessions."),
    ("Registrations", "COUNTROWS(registrations)", "#,0", "Number of registrations. One person registering for two sessions counts twice."),
    ("Attendees", "CALCULATE(COUNTROWS(registrations), registrations[attended] = TRUE())", "#,0", "Registrations where the person joined the session live."),
    ("Attendance Rate", "DIVIDE([Attendees], [Registrations])", "0.0%", "Share of registrations that turned into live attendance."),
    ("Total Capacity", "SUM(sessions[capacity])", "#,0", "Total seats available across sessions."),
    ("Fill Rate", "DIVIDE([Registrations], [Total Capacity])", "0.0%", "Registrations divided by capacity. 90% or more means the session is filling up."),
    ("Full Sessions", "COUNTROWS(FILTER(sessions, [Fill Rate] >= 0.9))", "#,0", "Sessions with a fill rate of 90% or more."),
    ("Avg Rating", "AVERAGE(feedback[rating])", "0.00", "Average feedback rating, from 1 (poor) to 5 (excellent)."),
    ("Responses", "COUNTROWS(feedback)", "#,0", "Number of feedback responses."),
]

with connect_semantic_model(dataset=MODEL, readonly=False) as tom:
    # Relationships (many-to-one, single direction)
    tom.add_relationship(from_table="registrations", from_column="session_id", to_table="sessions", to_column="session_id", from_cardinality="Many", to_cardinality="One")
    tom.add_relationship(from_table="feedback", from_column="session_id", to_table="sessions", to_column="session_id", from_cardinality="Many", to_cardinality="One")
    tom.add_relationship(from_table="sessions", from_column="speaker_id", to_table="speakers", to_column="speaker_id", from_cardinality="Many", to_cardinality="One")
    tom.add_relationship(from_table="sessions", from_column="session_date", to_table="dates", to_column="date", from_cardinality="Many", to_cardinality="One")
    tom.mark_as_date_table(table_name="dates", column_name="date")

    # Create/Update measures directly in table "sessions"
    for name, expr, fmt, desc in M:
        # add_measure normally does upsert (create or update if exists)
        tom.add_measure(
            table_name="sessions",
            measure_name=name,
            expression=expr,
            format_string=fmt,
            description=desc,
        )

# Refresh the model in the service
labs.refresh_semantic_model(dataset=MODEL)
print("Model ready:", MODEL)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 5. Answer key
# Keep this open during Act 1. These are the numbers a correct app shows. The model check underneath should print the same numbers.

# CELL ********************

k = answer_key(tables)
print(f"Sessions               {k['sessions']}")
print(f"Registrations          {k['registrations']:,}")
print(f"Attendees              {k['attendees']:,}")
print(f"Attendance rate        {k['attendance_rate']:.1%}")
print(f"Avg rating             {k['avg_rating']:.2f}  ({k['responses']:,} responses)")
print(f"Full sessions (90%+)   {k['sessions_over_90_fill']}")
print(f"Low-rated (<4, 10+ responses) {k['sessions_under_4']}")
print("\nBY LANGUAGE"); print(k["by_language"])
print("\nBY TRACK"); print(k["by_track"])
print("\nTOP 5 SESSIONS BY REGISTRATIONS"); print(k["top_sessions"])
print("\nLOWEST-RATED SESSIONS"); print(k["low_rated"])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

import sempy.fabric as fabric
display(fabric.evaluate_dax(MODEL, 'EVALUATE ROW("Sessions", [Sessions], "Registrations", [Registrations], "Attendance Rate", ROUND([Attendance Rate] * 100, 1), "Avg Rating", ROUND([Avg Rating], 2), "Full Sessions", [Full Sessions])'))
display(fabric.evaluate_dax(MODEL, 'EVALUATE SUMMARIZECOLUMNS(sessions[language], "Registrations", [Registrations], "Attendance Rate", ROUND([Attendance Rate] * 100, 1))'))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# MARKDOWN ********************

# ## 6. Optional: reset the demo
# Run this between rehearsals and after the session to remove the apps built live, so the next run starts clean. It only deletes items whose names are in the list below, and it never touches the lakehouse or the semantic model. **Leave `CONFIRM_RESET = False` unless you mean it.**
# 
# Backup apps end in `(backup)` and aren't in the list, so they survive a reset.
# 
# Apps built in Power BI Desktop for Apps also have a project folder on your PC. Delete or rename that folder by hand.

# CELL ********************

CONFIRM_RESET = False
DEMO_ITEM_NAMES = ["session-pulse", "idea-board", "lunch-picker", "kudos-wall", "scavenger-hunt"]

import sempy.fabric as fabric
items = fabric.list_items()
hits = items[items["Display Name"].isin(DEMO_ITEM_NAMES)]
print("Items that match the reset list:")
display(hits[["Display Name", "Type"]])
if CONFIRM_RESET:
    ws = fabric.get_workspace_id()
    client = fabric.FabricRestClient()
    for _, it in hits.iterrows():
        client.delete(f"v1/workspaces/{ws}/items/{it['Id']}")
        print(f"Deleted {it['Type']}: {it['Display Name']}")
else:
    print("Reset skipped. Check the list above, then set CONFIRM_RESET = True to delete those items.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
