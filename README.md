# Footvolley Analytics Pipeline ⚽🔥

מערכת ראייה ממוחשבת וניתוח נתונים אוטומטי למשחקי ומעגלי פוצ'יוולי, המבוססת על מודלי YOLOv8/YOLO11 ו-ByteTrack.

## 📌 סטטוס פרויקט (סיכום יום)
- **Perception Pipeline:** זיהוי שחקנים וכדור עובד ברמת דיוק של 80-90% על סרטוני ניסיון, תוך חילוץ נתונים מובנים ל-`tracking_data.json`.
- **ארגון הריפו:** ניקוי מלא של קבצי כדורסל ישנים וסידור תיקיית המודלים (`footvolley/models`).
- **אימות ויזואלי:** הוספת מנגנון המרה ל-`footvolley_preview.mp4` לצפייה ישירה ב-Codespaces.

## 🎯 יעדים קרובים (Post-Processing & Analytics)
1. **השלמת פערים (Linear Interpolation):** אלגוריתם להשלמת מיקומי כדור בפריימים שבהם הכדור נעלם לגובה/טשטוש.
2. **חלוקה לאזורים (Zonal Mapping):** שיוך דינמי של שחקנים לפי 4 אזורי מגרש (`Top`, `Bottom`, `Left`, `Right`).
3. **זיהוי נגיעות (Touch Detection):** שילוב קרבה מרחבית עם שינוי חד בווקטור המהירות של הכדור.

---
> **הערה למדדי הבנצ'מרק (`ball_v2`):**
> - סט הבדיקה הקיים כולל 40 פריימים מקליפ יחיד.
> - התוצאות מהוות ראיה כיוונית ראשונית בלבד.

## 👤 Credits & Contributions

**Original concept:** Ido - the project idea and vision.

**Technical foundation (as of November 3, 2026):** Built by Yuval Matyovis, including the player and ball detection and tracking pipeline, data preparation and training of the ball_v2 model, extraction of per-frame positions to tracking_data.json, and repository organization and documentation.

**Future contributions:** Each new contribution will be documented with the contributor's name, area of work, and a short description of the change, distinguishing the initial technical foundation from subsequent team contributions.
