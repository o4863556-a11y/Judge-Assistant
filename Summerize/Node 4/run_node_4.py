import json
import os
import sys
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Add parent directory to path for shared schema imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from node_4a import Node4A_ThematicClustering
from node_4b import Node4B_ThemeSynthesis

# Load environment variables (for GROQ_API_KEY)
load_dotenv()


# Sample Node 3 output for testing (two roles with mixed categories)
SAMPLE_INPUT = {
    "role_aggregations": [
        {
            "role": "الوقائع",
            "agreed": [
                {
                    "text": "أبرم الطرفان عقد بيع ابتدائي بتاريخ 5/6/2021 بشأن شقة سكنية بالعقار رقم 15 شارع النيل",
                    "sources": ["صحيفة دعوى.txt ص1 ف1", "مذكرة دفاع.txt ص1 ف1"]
                },
                {
                    "text": "تم الاتفاق على ثمن إجمالي قدره 500,000 جنيه مصري",
                    "sources": ["صحيفة دعوى.txt ص1 ف3", "مذكرة دفاع.txt ص1 ف2"]
                },
                {
                    "text": "سدد المدعي دفعة أولى بمبلغ 200,000 جنيه بموجب إيصال مؤرخ 5/6/2021",
                    "sources": ["صحيفة دعوى.txt ص2 ف1", "مذكرة دفاع.txt ص2 ف1"]
                },
                {
                    "text": "تم تسجيل العقد لدى الشهر العقاري بتاريخ 10/6/2021",
                    "sources": ["صحيفة دعوى.txt ص2 ف3"]
                },
            ],
            "disputed": [
                {
                    "subject": "مقدار المبالغ المسددة من الثمن",
                    "positions": [
                        {
                            "party": "المدعي",
                            "bullets": [
                                "تم سداد مبلغ 400,000 جنيه من إجمالي الثمن",
                                "الدفعة الثانية بمبلغ 200,000 جنيه سددت نقداً بتاريخ 1/9/2021"
                            ],
                            "sources": ["صحيفة دعوى.txt ص2 ف2", "صحيفة دعوى.txt ص3 ف1"]
                        },
                        {
                            "party": "المدعى عليه",
                            "bullets": [
                                "لم يتسلم سوى الدفعة الأولى بمبلغ 200,000 جنيه فقط",
                                "لا يوجد ما يثبت سداد أي مبالغ إضافية"
                            ],
                            "sources": ["مذكرة دفاع.txt ص2 ف2", "مذكرة دفاع.txt ص3 ف1"]
                        }
                    ]
                },
                {
                    "subject": "تاريخ التسليم المتفق عليه",
                    "positions": [
                        {
                            "party": "المدعي",
                            "bullets": ["الموعد المتفق عليه للتسليم هو 1/1/2022"],
                            "sources": ["صحيفة دعوى.txt ص3 ف2"]
                        },
                        {
                            "party": "المدعى عليه",
                            "bullets": ["لم يتضمن العقد موعداً محدداً للتسليم"],
                            "sources": ["مذكرة دفاع.txt ص3 ف2"]
                        }
                    ]
                },
                {
                    "subject": "حالة العقار عند المعاينة",
                    "positions": [
                        {
                            "party": "المدعي",
                            "bullets": ["العقار غير مطابق للمواصفات المتفق عليها ويوجد عيوب إنشائية"],
                            "sources": ["صحيفة دعوى.txt ص4 ف1"]
                        },
                        {
                            "party": "المدعى عليه",
                            "bullets": ["العقار سليم ومطابق وتم معاينته قبل التعاقد"],
                            "sources": ["مذكرة دفاع.txt ص4 ف1"]
                        }
                    ]
                },
            ],
            "party_specific": [
                {
                    "party": "المدعي",
                    "text": "يعاني المدعي من أضرار مادية جسيمة نتيجة التأخير في التسليم تقدر بمبلغ 50,000 جنيه",
                    "sources": ["صحيفة دعوى.txt ص4 ف2"]
                },
                {
                    "party": "المدعي",
                    "text": "اضطر المدعي لاستئجار مسكن بديل بإيجار شهري 3,000 جنيه منذ يناير 2022",
                    "sources": ["صحيفة دعوى.txt ص4 ف3"]
                },
                {
                    "party": "المدعى عليه",
                    "text": "تعرض المدعى عليه لظروف قاهرة تمثلت في ارتفاع أسعار مواد البناء بنسبة 40%",
                    "sources": ["مذكرة دفاع.txt ص4 ف2"]
                },
                {
                    "party": "المدعى عليه",
                    "text": "أخطر المدعى عليه المدعي بالتأخير بخطاب مسجل بتاريخ 15/11/2021",
                    "sources": ["مذكرة دفاع.txt ص5 ف1"]
                },
            ]
        },
        {
            "role": "الأساس القانوني",
            "agreed": [],
            "disputed": [
                {
                    "subject": "الأساس القانوني للفسخ",
                    "positions": [
                        {
                            "party": "المدعي",
                            "bullets": [
                                "يستند المدعي إلى المادة 418 من القانون المدني بشأن التزام البائع بنقل الملكية",
                                "يستند إلى المادة 157 بشأن الفسخ القضائي لعدم التنفيذ"
                            ],
                            "sources": ["صحيفة دعوى.txt ص5 ف1", "صحيفة دعوى.txt ص5 ف2"]
                        },
                        {
                            "party": "المدعى عليه",
                            "bullets": [
                                "يدفع بنص المادة 160 من القانون المدني بشأن انفساخ العقد بقوة القانون",
                                "يتمسك بالمادة 165 بشأن القوة القاهرة كسبب أجنبي"
                            ],
                            "sources": ["مذكرة دفاع.txt ص5 ف2", "مذكرة دفاع.txt ص5 ف3"]
                        }
                    ]
                },
            ],
            "party_specific": [
                {
                    "party": "المدعي",
                    "text": "يستند المدعي إلى حكم محكمة النقض في الطعن رقم 1234 لسنة 85 ق بشأن التزام البائع بالتسليم",
                    "sources": ["صحيفة دعوى.txt ص6 ف1"]
                },
                {
                    "party": "المدعي",
                    "text": "يحتج بالمادة 215 من القانون المدني بشأن التعويض عن عدم التنفيذ",
                    "sources": ["صحيفة دعوى.txt ص6 ف2"]
                },
                {
                    "party": "المدعى عليه",
                    "text": "يدفع بتوافر شروط القوة القاهرة وفقاً للمادة 373 من القانون المدني",
                    "sources": ["مذكرة دفاع.txt ص6 ف1"]
                },
                {
                    "party": "المدعى عليه",
                    "text": "يستند إلى مبدأ حسن النية في تنفيذ العقود وفقاً للمادة 148 مدني",
                    "sources": ["مذكرة دفاع.txt ص6 ف2"]
                },
            ]
        },
    ]
}


def main():
    # Check for API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\nWARNING: GROQ_API_KEY not found in environment variables.")
        print("Ensure you have a .env file or set the variable.")

    try:
        # Initialize LLM
        llm = ChatGroq(model_name="llama-3.3-70b-versatile")
        node_4a = Node4A_ThematicClustering(llm)
        node_4b = Node4B_ThemeSynthesis(llm)
    except Exception as e:
        print(f"Failed to initialize Node 4 or LLM: {e}")
        return

    # Use sample data or load from file if provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"Error: File not found at {file_path}")
            return
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                input_data = json.load(f)
        except Exception as e:
            print(f"Error loading input file: {e}")
            return
        print(f"\n--- Loaded input from: {file_path} ---")
    else:
        input_data = SAMPLE_INPUT
        print("\n--- Using sample Node 3 output ---")

    role_aggs = input_data.get("role_aggregations", [])
    total_items = 0
    for agg in role_aggs:
        n = (len(agg.get("agreed", []))
             + len(agg.get("disputed", []))
             + len(agg.get("party_specific", [])))
        total_items += n
    print(f"Input roles: {len(role_aggs)}, Total items: {total_items}")

    # --- Step 1: Node 4A (Thematic Clustering) ---
    print("\n" + "=" * 60)
    print("STEP 1: Thematic Clustering (Node 4A)")
    print("=" * 60)
    try:
        result_4a = node_4a.process(input_data)
    except Exception as e:
        print(f"Error during Node 4A processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print 4A results
    themed_roles = result_4a.get("themed_roles", [])
    if not themed_roles:
        print("No themed roles produced by Node 4A.")
        return

    for tr in themed_roles:
        role = tr.get("role", "?")
        themes = tr.get("themes", [])
        print(f"\n  Role: {role} -> {len(themes)} theme(s)")
        for t in themes:
            print(f"    - {t.get('theme_name', '?')} "
                  f"(agreed={len(t.get('agreed', []))}, "
                  f"disputed={len(t.get('disputed', []))}, "
                  f"party_specific={len(t.get('party_specific', []))})")

    # Save 4A output
    output_4a_path = os.path.join(os.path.dirname(__file__), "node_4a_output.json")
    try:
        with open(output_4a_path, "w", encoding="utf-8") as f:
            json.dump(result_4a, f, ensure_ascii=False, indent=2)
        print(f"\n4A output saved to: {output_4a_path}")
    except Exception as e:
        print(f"Warning: Could not save 4A output file: {e}")

    # --- Step 2: Node 4B (Theme-Level Synthesis) ---
    print("\n" + "=" * 60)
    print("STEP 2: Theme-Level Synthesis (Node 4B)")
    print("=" * 60)
    try:
        result_4b = node_4b.process(result_4a)
    except Exception as e:
        print(f"Error during Node 4B processing: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print 4B results
    role_summaries = result_4b.get("role_theme_summaries", [])
    if not role_summaries:
        print("No role theme summaries produced by Node 4B.")
        return

    print(f"\n=== Theme Summaries ({len(role_summaries)} roles) ===")
    for rs in role_summaries:
        role = rs.get("role", "?")
        summaries = rs.get("theme_summaries", [])
        print(f"\n{'=' * 60}")
        print(f"Role: {role} ({len(summaries)} theme summaries)")

        for i, ts in enumerate(summaries, 1):
            print(f"\n  [{i}] Theme: {ts.get('theme', '?')}")
            print(f"      Key Disputes: {ts.get('key_disputes', [])}")
            print(f"      Sources: {ts.get('sources', [])}")
            summary = ts.get("summary", "")
            # Show first 300 chars of summary for readability
            if len(summary) > 300:
                print(f"      Summary: {summary[:300]}...")
            else:
                print(f"      Summary: {summary}")

        print("-" * 60)

    # Save 4B output
    output_4b_path = os.path.join(os.path.dirname(__file__), "node_4b_output.json")
    try:
        with open(output_4b_path, "w", encoding="utf-8") as f:
            json.dump(result_4b, f, ensure_ascii=False, indent=2)
        print(f"\nFinal output saved to: {output_4b_path}")
    except Exception as e:
        print(f"Warning: Could not save 4B output file: {e}")


if __name__ == "__main__":
    main()
