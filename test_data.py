import sys
from pathlib import Path
import json
import random
import uuid
import numpy as np
import faiss
import datetime
import shutil
import textwrap
import pytz

# --- Path Setup ---
# Add the project root to the Python path to allow importing from 'src'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.config_loader import get_config, PROJECT_ROOT
    from src.logger_setup import setup_logging, logger
except ImportError as e:
    print("\nERROR: Could not import necessary Samson modules.")
    print("Please run this script from the project's root directory (e.g., 'python tools/setup_test_data.py').")
    print(f"Details: {e}")
    sys.exit(1)

# --- Configuration ---
# --- START MODIFICATION: Use predefined data ---
# Set NUM_SPEAKERS based on the provided data
NUM_SPEAKERS = 3
NUM_MATTERS = 4
NUM_DAYS_OF_DATA = 7
NUM_CHUNKS_PER_DAY_RANGE = (4, 8)
PROBABILITY_NO_MATTER = 0.25

# --- Predefined Data Content ---

PREDEFINED_SPEAKER_PROFILES = [
  {
    "faiss_id": 0,
    "name": "Partner",
    "role": None,
    "created_utc": "2025-09-24T20:58:47.151021+00:00",
    "last_updated_utc": "2025-09-24T20:58:47.263153+00:00",
    "llm_summary_all_dialogue": None,
    "last_role_inference_utc": None,
    "profile_last_evolved_utc": None,
    "dynamic_threshold_feedback": [
      {
        "timestamp_utc": "2025-09-24T20:58:47.263012+00:00",
        "correction_type": "flag_resolution",
        "original_speaker_id": "CUSID_20250924_62955_0",
        "corrected_speaker_id": "Partner",
        "source": "cockpit_gui_flag_review",
        "audio_context": "in_person",
        "chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb",
        "duration_s": 545.0399999999945,
        "flag_details": {
          "flag_id": "FLAG_20250924_6f2b0fc962f3",
          "reason_for_flag": "New unknown speaker detected.",
          "flag_type": None
        }
      }
    ],
    "segment_embeddings_for_evolution": {
      "in_person": [
        {
          "embedding": [
            -0.02940201945602894,
            -0.009524257853627205,
            0.012674396857619286,
            -0.13303539156913757,
            0.09708263725042343,
            -0.12714171409606934,
            -0.017889874055981636,
            -0.15489907562732697,
            0.006093460600823164,
            0.047320593148469925,
            -0.0663822740316391,
            0.05805978178977966,
            0.13142424821853638,
            0.04911082237958908,
            -0.04556604474782944,
            0.0374559722840786,
            -0.010513841174542904,
            0.037549279630184174,
            0.07081723213195801,
            0.0029871852602809668,
            -0.05163031071424484,
            -0.04339450225234032,
            0.14682736992835999,
            0.001753709395416081,
            0.07832057774066925,
            -0.08101241290569305,
            0.05675743520259857,
            -0.03261455148458481,
            0.0007419526227749884,
            -0.02919701673090458,
            -0.034999821335077286,
            0.0007348551298491657,
            0.006818156689405441,
            0.17608265578746796,
            -0.014996602199971676,
            -0.06169426441192627,
            0.11074592918157578,
            -0.05911023169755936,
            0.0161677785217762,
            -0.0314270444214344,
            -0.05078626796603203,
            0.05241452530026436,
            0.0948924571275711,
            0.020148789510130882,
            0.13902880251407623,
            -0.010452479124069214,
            0.006574113853275776,
            -0.06599297374486923,
            0.0900496318936348,
            -0.13078834116458893,
            0.0511520653963089,
            0.08154360204935074,
            -0.07202230393886566,
            -0.12023188918828964,
            -0.008893853984773159,
            0.002145694801583886,
            -0.05723682418465614,
            -0.1708788424730301,
            -0.06308043003082275,
            0.19251397252082825,
            -0.0021972591057419777,
            0.059173449873924255,
            0.0003399199922569096,
            0.06661970168352127,
            0.08232112973928452,
            0.12487239390611649,
            0.055835701525211334,
            -0.10066651552915573,
            0.0308330450206995,
            -0.07836435735225677,
            0.07936213910579681,
            0.05929189547896385,
            -0.06773647665977478,
            0.010329904034733772,
            0.04557317495346069,
            0.00037787947803735733,
            -0.006722146179527044,
            -0.08744056522846222,
            -0.021841147914528847,
            0.03385632857680321,
            0.014234364964067936,
            0.02709072455763817,
            0.0911131352186203,
            -0.1080314964056015,
            0.00454972917214036,
            -0.002611207775771618,
            -0.05204955115914345,
            0.11730774492025375,
            0.048628345131874084,
            0.15291085839271545,
            -0.015821656212210655,
            -0.08079210668802261,
            -0.09316851198673248,
            0.08424917608499527,
            -0.02644588053226471,
            0.11381960660219193,
            0.010838204994797707,
            -0.01559568289667368,
            0.009977675043046474,
            0.03979561850428581,
            -0.10037596523761749,
            -0.11585795879364014,
            0.03637973219156265,
            0.10872776806354523,
            0.01611967757344246,
            -0.023984059691429138,
            0.032237861305475235,
            -0.11730100214481354,
            0.06164667382836342,
            0.011561512015759945,
            -0.13284333050251007,
            -0.04473241791129112,
            0.031645242124795914,
            -0.07257834076881409,
            -0.002702531637623906,
            0.1721053272485733,
            -0.15498633682727814,
            0.07048940658569336,
            0.035259559750556946,
            0.00909479334950447,
            0.020720109343528748,
            0.1603063941001892,
            0.013397309929132462,
            -0.06706787645816803,
            0.03882654011249542,
            -0.01303553394973278,
            0.11481355130672455,
            -0.02406533621251583,
            0.06151469051837921,
            -0.04753124713897705,
            -0.05692601576447487,
            0.0036661040503531694,
            0.0007489419658668339,
            -0.018574699759483337,
            0.021748360246419907,
            -0.06616009771823883,
            -0.041512493044137955,
            0.03329518437385559,
            -0.03204836696386337,
            -0.08921577036380768,
            -0.1454656720161438,
            -0.03288769721984863,
            -0.08165831863880157,
            -0.02226983942091465,
            -0.003756695194169879,
            0.06444954872131348,
            0.10431332141160965,
            -0.006848582066595554,
            0.01802101545035839,
            0.0967739000916481,
            -0.03542456403374672,
            0.03316819667816162,
            0.13536697626113892,
            -0.0016227983869612217,
            -0.02193465270102024,
            -0.04002697020769119,
            0.08187063783407211,
            -0.015762031078338623,
            0.08139394223690033,
            -0.008462770842015743,
            -0.03118753246963024,
            -0.0447993278503418,
            -0.09501964598894119,
            0.05759914964437485,
            -0.03398936614394188,
            0.09841006249189377,
            -0.027602050453424454,
            0.04027586430311203,
            0.057633284479379654,
            0.015506481751799583,
            0.07010962814092636,
            0.08620932698249817,
            0.03511127084493637,
            -0.00787417683750391,
            -0.062495648860931396,
            -0.09117912501096725,
            0.04374849796295166,
            -0.041479744017124176,
            0.027325384318828583,
            -0.0419960618019104,
            0.04416925460100174,
            -0.1426422894001007,
            -0.031758878380060196,
            -0.02269287034869194,
            0.06535381078720093,
            0.040340274572372437,
            -0.0599735751748085,
            -0.11990807205438614,
            0.007505821995437145,
            -0.16985537111759186,
            0.06256645917892456,
            0.011407455429434776
          ],
          "duration_s": 10.24,
          "diarization_confidence": 1.0,
          "timestamp_utc": "2025-09-24T20:58:47.258905+00:00",
          "source_chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb"
        }
      ]
    },
    "associated_matter_ids": [],
    "lifetime_total_audio_s": 545.0399999999945,
    "speaker_relationships": {}
  },
  {
    "faiss_id": 1,
    "name": "System Administrator",
    "role": None,
    "created_utc": "2025-09-24T20:58:55.220276+00:00",
    "last_updated_utc": "2025-09-24T20:58:55.316357+00:00",
    "llm_summary_all_dialogue": None,
    "last_role_inference_utc": None,
    "profile_last_evolved_utc": None,
    "dynamic_threshold_feedback": [
      {
        "timestamp_utc": "2025-09-24T20:58:55.315858+00:00",
        "correction_type": "flag_resolution",
        "original_speaker_id": "CUSID_20250924_62955_1",
        "corrected_speaker_id": "System Administrator",
        "source": "cockpit_gui_flag_review",
        "audio_context": "in_person",
        "chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb",
        "duration_s": 304.95999999999754,
        "flag_details": {
          "flag_id": "FLAG_20250924_f9ee1eaf022e",
          "reason_for_flag": "New unknown speaker detected.",
          "flag_type": None
        }
      }
    ],
    "segment_embeddings_for_evolution": {
      "in_person": [
        {
          "embedding": [
            0.08801856637001038,
            0.16245707869529724,
            0.027313046157360077,
            -0.14914007484912872,
            0.06303966790437698,
            0.07345685362815857,
            0.04630923643708229,
            -0.041902389377355576,
            -0.06476744264364243,
            -0.04108376428484917,
            -0.05806145444512367,
            0.029864422976970673,
            -0.010928685776889324,
            0.03936200588941574,
            0.023980949074029922,
            -0.07733345031738281,
            0.06948164105415344,
            0.008302946574985981,
            0.13210134208202362,
            -0.052547525614500046,
            -0.02807171270251274,
            -0.1465173214673996,
            -0.0007869324763305485,
            0.08773764222860336,
            0.08314559608697891,
            0.052916742861270905,
            -0.022583365440368652,
            0.0909699872136116,
            -0.02275511622428894,
            -0.06840100139379501,
            0.15846356749534607,
            0.12140806019306183,
            -0.017901930958032608,
            -0.09097365289926529,
            0.050178252160549164,
            0.05398018658161163,
            0.01240605115890503,
            0.034771233797073364,
            0.18864233791828156,
            0.042731210589408875,
            0.10156184434890747,
            0.048036254942417145,
            0.011851799674332142,
            0.09077923744916916,
            -0.06410378217697144,
            -0.034124381840229034,
            -0.1539764404296875,
            -0.061636049300432205,
            -0.04894208908081055,
            -0.059292230755090714,
            -0.02752644754946232,
            0.0893697664141655,
            -0.00556194270029664,
            0.002871658420190215,
            0.08709285408258438,
            0.01729617826640606,
            -0.10577315092086792,
            -0.05200355127453804,
            0.03696824610233307,
            -0.030224589630961418,
            0.09259915351867676,
            0.0071981679648160934,
            0.11265040934085846,
            0.08622093498706818,
            -0.029195494949817657,
            0.03606683015823364,
            0.004755625035613775,
            0.07362145930528641,
            -0.00188703543972224,
            -0.018081096932291985,
            -0.08052345365285873,
            -0.10150366276502609,
            -0.09641866385936737,
            -0.1246107816696167,
            -0.06365887820720673,
            -0.00171032571233809,
            -0.052077457308769226,
            0.01741557940840721,
            0.06370792537927628,
            0.08372274041175842,
            0.009242244996130466,
            -0.10078471153974533,
            0.0052657825872302055,
            -0.07334616780281067,
            -0.042036913335323334,
            0.06207882985472679,
            0.10973308235406876,
            0.0948476791381836,
            -0.024651579558849335,
            0.07364634424448013,
            0.006122741848230362,
            0.013353477232158184,
            0.01169854961335659,
            0.0009011916117742658,
            -0.071079783141613,
            -0.03595786169171333,
            -0.09406022727489471,
            -0.0006619549822062254,
            -0.03872820362448692,
            0.05346975103020668,
            -0.13563434779644012,
            -0.015386275947093964,
            -0.0005265066865831614,
            0.08933202177286148,
            -0.00022586061095353216,
            -0.1064811646938324,
            -0.11292620748281479,
            -0.06297257542610168,
            -0.01254722848534584,
            0.06452403962612152,
            0.058936040848493576,
            -0.030214868485927582,
            -0.007063334807753563,
            -0.05292568728327751,
            -0.08124188333749771,
            0.10717770457267761,
            0.035750873386859894,
            0.10637405514717102,
            0.033960022032260895,
            -0.05649297311902046,
            0.06611819565296173,
            0.03169861063361168,
            -0.05875557288527489,
            -0.02048330381512642,
            -0.02506474405527115,
            -0.1148657277226448,
            0.09347926080226898,
            -0.011704836040735245,
            0.00223794998601079,
            -0.09645381569862366,
            -0.01857141964137554,
            0.08057820796966553,
            -0.0019904046785086393,
            -0.13948707282543182,
            0.036027081310749054,
            -0.055357322096824646,
            -0.07251962274312973,
            -0.10367660224437714,
            0.09885543584823608,
            -0.05335216224193573,
            0.003224943997338414,
            -0.05278104916214943,
            -0.015929358080029488,
            -0.062317751348018646,
            -0.016946038231253624,
            0.032666463404893875,
            0.05758725106716156,
            -0.022715827450156212,
            -0.05144856497645378,
            0.06138624623417854,
            0.0005735569866374135,
            0.08916320651769638,
            0.0026378524489700794,
            0.05133848637342453,
            -0.039284221827983856,
            -0.0453856959939003,
            -0.010526898317039013,
            0.11293839663267136,
            -0.057234298437833786,
            0.09392478317022324,
            0.03859010338783264,
            -0.010574846528470516,
            -0.0825679674744606,
            -0.06902945041656494,
            -0.004404135514050722,
            -0.16427843272686005,
            0.0061039249412715435,
            0.03654037415981293,
            0.03260938078165054,
            -0.03267056122422218,
            0.051829174160957336,
            0.24068181216716766,
            0.06716202944517136,
            0.09607525169849396,
            -0.027484320104122162,
            -0.14114372432231903,
            0.07164543867111206,
            0.16356246173381805,
            -0.03891809284687042,
            0.03777867555618286,
            0.01258552260696888,
            -0.0540601871907711,
            -0.11766959726810455,
            0.02471057139337063,
            -0.09203092753887177,
            -0.053660985082387924,
            -0.09708276391029358,
            -0.03454280272126198,
            0.01194962952286005,
            -0.029865646734833717,
            -0.002296813065186143,
            -0.09497830271720886
          ],
          "duration_s": 12.0,
          "diarization_confidence": 1.0,
          "timestamp_utc": "2025-09-24T20:58:55.312016+00:00",
          "source_chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb"
        }
      ]
    },
    "associated_matter_ids": [],
    "lifetime_total_audio_s": 304.95999999999754,
    "speaker_relationships": {}
  },
  {
    "faiss_id": 2,
    "name": "Senior Associate",
    "role": None,
    "created_utc": "2025-09-24T20:59:05.385958+00:00",
    "last_updated_utc": "2025-09-24T20:59:05.624585+00:00",
    "llm_summary_all_dialogue": None,
    "last_role_inference_utc": None,
    "profile_last_evolved_utc": None,
    "dynamic_threshold_feedback": [
      {
        "timestamp_utc": "2025-09-24T20:59:05.623990+00:00",
        "correction_type": "flag_resolution",
        "original_speaker_id": "CUSID_20250924_62955_2",
        "corrected_speaker_id": "Senior Associate",
        "source": "cockpit_gui_flag_review",
        "audio_context": "in_person",
        "chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb",
        "duration_s": 325.67999999999824,
        "flag_details": {
          "flag_id": "FLAG_20250924_b762eafd9e54",
          "reason_for_flag": "New unknown speaker detected.",
          "flag_type": None
        }
      }
    ],
    "segment_embeddings_for_evolution": {
      "in_person": [
        {
          "embedding": [
            -0.020878611132502556,
            0.042755283415317535,
            0.10753796994686127,
            0.06109590828418732,
            0.12148942798376083,
            -0.00892389751970768,
            -0.10774990916252136,
            0.04677440598607063,
            -0.10175977647304535,
            0.10180628299713135,
            0.00838007777929306,
            0.07971809804439545,
            0.03378855437040329,
            -0.10030122101306915,
            0.038622479885816574,
            0.057512879371643066,
            0.12253731489181519,
            0.0583689846098423,
            0.10025814920663834,
            -0.014348807744681835,
            -0.01750672049820423,
            -0.06746197491884232,
            -0.00011906291911145672,
            0.09558454900979996,
            -0.13521692156791687,
            0.07402805238962173,
            0.02885361947119236,
            -0.023380571976304054,
            0.18725986778736115,
            -0.03845212981104851,
            -0.007171970326453447,
            0.010604779236018658,
            0.007802046835422516,
            -0.04641370475292206,
            0.0015690657310187817,
            0.02314179763197899,
            0.012879254296422005,
            -0.00757969543337822,
            -0.18021652102470398,
            -0.06996295601129532,
            -0.07934241741895676,
            -0.02532757818698883,
            0.10006073862314224,
            -0.03433842584490776,
            0.0487278550863266,
            0.09371621161699295,
            -0.005959194619208574,
            -0.04550796002149582,
            0.12279891222715378,
            0.08409617096185684,
            -0.052349526435136795,
            0.043459050357341766,
            -0.06970586627721786,
            -0.03677184507250786,
            -0.1230543851852417,
            -0.04935029149055481,
            0.02482098713517189,
            -0.030188895761966705,
            -0.014236836694180965,
            0.03470458462834358,
            -0.08057592064142227,
            0.05958893150091171,
            -0.06920500844717026,
            0.023551514372229576,
            -0.05412038043141365,
            0.0015016705729067326,
            -0.028013061732053757,
            -0.0700986459851265,
            -0.1384728103876114,
            -0.06728267669677734,
            -0.12360699474811554,
            -0.10132880508899689,
            -0.019144723191857338,
            -0.021338287740945816,
            0.0802798867225647,
            0.021517038345336914,
            0.05985730141401291,
            -0.020232580602169037,
            0.06276451051235199,
            0.09020435065031052,
            0.04185621812939644,
            0.0218562800437212,
            -0.06955790519714355,
            -0.02199595980346203,
            0.08168668299913406,
            -0.014144842512905598,
            0.08466894179582596,
            -0.022132646292448044,
            -0.05437086150050163,
            0.01779142953455448,
            0.03903064504265785,
            0.029121993109583855,
            0.0921819657087326,
            0.009185748174786568,
            7.068858394632116e-05,
            0.04865064099431038,
            -0.07547542452812195,
            -0.052293483167886734,
            0.013534588739275932,
            0.08425373584032059,
            -0.0905810222029686,
            -0.12974637746810913,
            0.01825585961341858,
            0.059347182512283325,
            -0.11461325734853745,
            -0.002448785351589322,
            0.0270393006503582,
            0.1009102538228035,
            -0.1167670264840126,
            0.07417524605989456,
            -0.05435667186975479,
            0.022181827574968338,
            0.07600969821214676,
            0.042595889419317245,
            0.011664041317999363,
            0.0632685124874115,
            -0.06300810724496841,
            0.06771016120910645,
            0.0376940481364727,
            0.0032169094774872065,
            0.0017603331943973899,
            -0.15201203525066376,
            -0.03199709579348564,
            0.05031052976846695,
            0.03779701143503189,
            0.038962315768003464,
            -0.03374180197715759,
            0.0692293718457222,
            -0.10638032108545303,
            -0.03125881031155586,
            -0.04725094139575958,
            0.06929048150777817,
            -0.12077783793210983,
            0.05718831717967987,
            -0.08137091994285583,
            -0.04990850389003754,
            -0.09377985447645187,
            -0.010951167903840542,
            -0.036551088094711304,
            -0.10230088233947754,
            0.002693494316190481,
            -0.007169794291257858,
            -0.011906263418495655,
            -0.09540869295597076,
            0.0627223327755928,
            0.0060302577912807465,
            0.04198077321052551,
            0.027874410152435303,
            -0.06316546350717545,
            0.21773944795131683,
            -0.030787993222475052,
            0.14289945363998413,
            -0.0723140686750412,
            0.020747225731611252,
            -0.017305472865700722,
            0.030873170122504234,
            0.0004897556500509381,
            -0.09478746354579926,
            -0.006087626330554485,
            -0.07887239009141922,
            -0.02333945594727993,
            -0.15576425194740295,
            0.05846777930855751,
            0.008716545067727566,
            -0.01930255815386772,
            -0.03994734212756157,
            0.012398803606629372,
            0.07028694450855255,
            -0.00814744271337986,
            -0.07691548764705658,
            -0.02877231501042843,
            0.06389909237623215,
            -0.051148656755685806,
            0.04918478801846504,
            0.053611453622579575,
            -0.16804547607898712,
            -0.02261391468346119,
            0.033778827637434006,
            0.11693775653839111,
            0.093439482152462,
            -0.06310394406318665,
            -0.034785736352205276,
            0.04076756164431572,
            0.10822153836488724,
            0.13155409693717957,
            0.08784548193216324,
            0.12317881733179092,
            -0.049167752265930176,
            -0.10230714827775955,
            0.13122513890266418,
            0.12888920307159424,
            -0.08333521336317062
          ],
          "duration_s": 11.999999999999998,
          "diarization_confidence": 1.0,
          "timestamp_utc": "2025-09-24T20:59:05.614810+00:00",
          "source_chunk_id": "4942534b-47db-439a-a0dc-1eac1d1a4bbb"
        }
      ]
    },
    "associated_matter_ids": [],
    "lifetime_total_audio_s": 325.67999999999824,
    "speaker_relationships": {}
  }
]

# Map the fake matter content to the specific IDs from the speaker profiles
PREDEFINED_MATTERS = [
    {"id": "m_e9e14b5c", "name": "Project Chimera", "desc": "Investigation into the anomalous energy readings from sub-level 7.", "keywords": ["energy", "anomaly", "sub-level 7", "containment"]},
    {"id": "m_95b4db7c", "name": "Quantum Entanglement Comms", "desc": "Developing and testing the FTL communicator prototype.", "keywords": ["FTL", "communication", "quantum", "prototype", "lag"]},
    {"id": "m_651c41bb", "name": "Xenobiology Analysis (Specimen 3B)", "desc": "Analyzing the cellular regeneration of Specimen 3B.", "keywords": ["xenobiology", "specimen 3B", "regeneration", "biopsy"]},
    # This ID appears in System Administrator's profile but not others. We'll map it to the 4th matter.
    {"id": "m_604321d3", "name": "Site Maintenance Review", "desc": "Weekly review of ongoing site maintenance and resource allocation.", "keywords": ["maintenance", "logistics", "power grid", "repairs"]}
]

LOREM_IPSUM_WORDS = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum".split()
# --- END MODIFICATION ---

def get_samson_today(config):
    """
    Gets the current date based on the 'assumed_recording_timezone' from the config.
    This is a standalone implementation for this script.
    """
    tz_str = config['timings'].get('assumed_recording_timezone', 'UTC')
    try:
        target_tz = pytz.timezone(tz_str)
    except pytz.UnknownTimeZoneError:
        logger.warning(f"Unknown timezone '{tz_str}' in config. Falling back to UTC.")
        target_tz = pytz.utc
    now_utc = datetime.datetime.now(pytz.utc)
    today_in_tz = now_utc.astimezone(target_tz)
    return today_in_tz.date()

# --- Main Generator Class ---

class TestDataGenerator:
    def __init__(self, config):
        self.config = config
        self.paths = config['paths']
        self.audio_suite_settings = config['audio_suite_settings']
        self.timings = config['timings']
        self.context_management = config['context_management']
        self.embedding_dim = self.audio_suite_settings.get('embedding_dim', 192)
        self.speakers = []
        self.matters = []
        self.generated_chunk_ids = []

    def confirm_and_clean_directories(self):
        """Asks for user confirmation and cleans relevant data directories."""
        print("--- Samson Test Data Generator ---")
        print("\nThis script will generate fake data for testing the Samson Cockpit GUI.")
        print("It will DELETE existing data in the following configured directories:")
        
        task_data_path = Path(self.config.get('task_intelligence', {}).get('task_data_file', 'data/tasks/tasks.jsonl'))
        
        dirs_to_clean = [
            self.paths['speaker_db_dir'],
            self.paths['database_folder'],
            self.paths['daily_log_folder'],
            self.paths['archived_audio_folder'],
            self.paths['flags_queue_dir'],
            task_data_path.parent
        ]

        for dir_path in dirs_to_clean:
            print(f"  - {dir_path}")

        response = input("\nARE YOU SURE you want to proceed? (yes/no): ").lower()
        if response != 'yes':
            print("Operation cancelled.")
            sys.exit(0)

        for dir_path in dirs_to_clean:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print("\nCleaned directories successfully.")

    def generate_speakers_and_matters(self):
        """Generates speaker profiles, FAISS index, speaker map, and matters.json from PREDEFINED data."""
        logger.info("--- Generating Speakers and Matters from Predefined Data---")
        
        # --- START MODIFICATION: Generate Matters from predefined list ---
        for matter_data in PREDEFINED_MATTERS:
            self.matters.append({
                "matter_id": matter_data["id"],
                "name": matter_data["name"],
                "description": matter_data["desc"],
                "keywords": matter_data["keywords"],
                "status": "active",
                "source": "test_data_generator"
            })
        # --- END MODIFICATION ---

        matters_path = self.paths['speaker_db_dir'] / "matters.json"
        with open(matters_path, 'w') as f:
            json.dump(self.matters, f, indent=2)
        logger.info(f"Generated {len(self.matters)} matters in '{matters_path}'.")

        # --- START MODIFICATION: Use predefined speakers and generate FAISS from their embeddings ---
        self.speakers = PREDEFINED_SPEAKER_PROFILES
        speaker_map = {}
        faiss_embeddings = []

        for speaker_profile in self.speakers:
            # Speaker Map
            speaker_map[speaker_profile['faiss_id']] = {"name": speaker_profile['name'], "context": "in_person"} # Assume in_person context

            # Embedding for FAISS index
            all_speaker_embeddings = []
            evo_data = speaker_profile.get("segment_embeddings_for_evolution", {})
            if evo_data:
                for context_type, embedding_list in evo_data.items():
                    for item in embedding_list:
                        if 'embedding' in item and item['embedding']:
                            all_speaker_embeddings.append(np.array(item['embedding']))
            
            if all_speaker_embeddings:
                # Calculate the mean embedding to represent the speaker
                mean_embedding = np.mean(all_speaker_embeddings, axis=0).astype('float32')
                mean_embedding /= np.linalg.norm(mean_embedding)
                faiss_embeddings.append(mean_embedding)
                logger.info(f"Created FAISS vector for '{speaker_profile['name']}' from {len(all_speaker_embeddings)} provided embeddings.")
            else:
                # Fallback to a random embedding if none are provided
                embedding = np.random.rand(self.embedding_dim).astype('float32')
                embedding /= np.linalg.norm(embedding)
                faiss_embeddings.append(embedding)
                logger.warning(f"No embeddings found for '{speaker_profile['name']}'. Using a random vector for FAISS index.")

        # Save Speaker Profiles
        profiles_path = self.paths['speaker_db_dir'] / "speaker_profiles.json"
        with open(profiles_path, 'w') as f:
            json.dump(self.speakers, f, indent=2)
        logger.info(f"Saved {len(self.speakers)} predefined speaker profiles in '{profiles_path}'.")
        
        # Save Speaker Map
        map_path = self.paths['speaker_db_dir'] / self.audio_suite_settings['speaker_map_filename']
        with open(map_path, 'w') as f:
            json.dump(speaker_map, f, indent=2)
        logger.info(f"Generated speaker map with {len(speaker_map)} entries in '{map_path}'.")

        # Save FAISS Index
        index = faiss.IndexFlatIP(self.embedding_dim)
        index.add(np.array(faiss_embeddings))
        index_path = self.paths['speaker_db_dir'] / self.audio_suite_settings['faiss_index_filename']
        faiss.write_index(index, str(index_path))
        logger.info(f"Generated FAISS index with {index.ntotal} vectors in '{index_path}'.")
        # --- END MODIFICATION ---


    def generate_persistent_state_files(self):
        """Generates context.json and events.jsonl with sample data."""
        logger.info("--- Generating Persistent State Files (Context and Events) ---")
        data_dir = PROJECT_ROOT / "data"
        data_dir.mkdir(exist_ok=True)

        # 1. Create context.json
        context_data = {
            "matter_id": None,
            "matter_name": None,
            "source": "test_data_generator",
            "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()
        }
        context_path = data_dir / "context.json"
        with open(context_path, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2)
        logger.info(f"Generated a null initial context in '{context_path}' (no active matter).")

        # 2. Create an empty events.jsonl to ensure no queued matter changes
        events_path = data_dir / "events.jsonl"
        events_path.touch()
        logger.info(f"Generated an empty events file in '{events_path}' to ensure no queued matter changes.")


    def generate_daily_data(self):
        """Generates daily logs, flags, archived audio, and master transcripts."""
        logger.info(f"--- Generating Daily Data for the Past {NUM_DAYS_OF_DATA} Days ---")
        
        today = get_samson_today(self.config)
        
        for i in range(1, NUM_DAYS_OF_DATA + 1):
            current_date = today - datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Generating data for {date_str}...")

            # Setup for the day
            assumed_tz_str = self.timings.get('assumed_recording_timezone', 'UTC')
            local_tz = pytz.timezone(assumed_tz_str)
            day_start_time_local = local_tz.localize(datetime.datetime.combine(current_date, datetime.time(9, 0, 0)))
            day_start_time_utc = day_start_time_local.astimezone(datetime.timezone.utc)

            daily_log_chunks = {}
            daily_flags = []
            all_words_for_master_log = []
            
            num_chunks = random.randint(NUM_CHUNKS_PER_DAY_RANGE[0], NUM_CHUNKS_PER_DAY_RANGE[1])
            
            for chunk_seq in range(1, num_chunks + 1):
                chunk_id = f"chunk_{date_str.replace('-', '')}_{uuid.uuid4().hex[:8]}"
                self.generated_chunk_ids.append(chunk_id)
                chunk_start_utc = day_start_time_utc + datetime.timedelta(minutes=2 * (chunk_seq - 1))
                
                # Create fake archived audio file
                archive_date_dir = self.paths['archived_audio_folder'] / date_str
                archive_date_dir.mkdir(parents=True, exist_ok=True)
                audio_filename = f"alibi-recording-audio_recordings-{chunk_seq}.aac"
                (archive_date_dir / audio_filename).touch()
                
                nested_segments, matter_segments = self._generate_dialogue_for_chunk(chunk_start_utc)
                flat_word_list = [word for segment in nested_segments for word in segment.get('words', [])]
                all_words_for_master_log.extend(flat_word_list)
                
                # --- START MODIFICATION: Disable flag generation ---
                if False: # Create a speaker ambiguity flag
                    daily_flags.append(self._create_speaker_flag(flat_word_list, audio_filename, chunk_id, date_str))

                if False: # Create a matter ambiguity flag
                    matter_flag = self._create_matter_flag(flat_word_list, audio_filename, chunk_id, date_str)
                    if matter_flag:
                        daily_flags.append(matter_flag)
                # --- END MODIFICATION ---

                daily_log_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "entry_creation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "original_file_name": audio_filename,
                    "file_sequence_number": chunk_seq,
                    "audio_chunk_start_utc": chunk_start_utc.isoformat(),
                    "matter_segments": matter_segments,
                    "processed_data": {
                        "word_level_transcript_with_absolute_times": nested_segments,
                    },
                    "source_job_output_dir": f"/fake/job/dir/{audio_filename}",
                    "processing_date_utc": date_str
                }

            # Save daily log file.
            log_path = self.paths['daily_log_folder'] / f"{date_str}_samson_log.json"
            daily_log_content = {
                "schema_version": "2.0",
                "day_start_timestamp_utc": day_start_time_utc.isoformat(),
                "chunks": daily_log_chunks,
                "matters": self.matters
            }
            with open(log_path, 'w') as f:
                json.dump(daily_log_content, f, indent=2)

            # Save daily flags to their own correct file.
            if daily_flags:
                flags_queue_path = self.paths['flags_queue_dir'] / f"{date_str}_flags_queue.json"
                with open(flags_queue_path, 'w') as f:
                    json.dump(daily_flags, f, indent=2)
                logger.info(f"Generated {len(daily_flags)} flags in '{flags_queue_path.name}'.")
            else:
                logger.info("Flag generation is disabled, no flags created.")

            
            # Generate Master Log
            self._generate_master_log(all_words_for_master_log, day_start_time_utc, current_date)

    def _generate_dialogue_for_chunk(self, chunk_start_utc):
        all_segments = []
        matter_segments = []
        current_time = 0.0
        
        num_turns = random.randint(5, 15)
        
        last_speaker_name = None
        last_matter_id = None
        
        for _ in range(num_turns):
            speaker = random.choice([s for s in self.speakers if s['name'] != last_speaker_name])
            last_speaker_name = speaker['name']
            
            if random.random() < PROBABILITY_NO_MATTER:
                current_matter_id = None
            else:
                matter = random.choice(self.matters)
                current_matter_id = matter['matter_id']

            if current_matter_id != last_matter_id:
                if matter_segments:
                    matter_segments[-1]['end_time'] = current_time
                
                if current_matter_id is not None:
                    matter_segments.append({
                        "start_time": current_time,
                        "end_time": -1,
                        "matter_id": current_matter_id
                    })

            words_for_this_turn = []
            turn_text_parts = []
            turn_start_time = current_time
            
            num_words = random.randint(5, 25)
            for _ in range(num_words):
                word_text = random.choice(LOREM_IPSUM_WORDS)
                turn_text_parts.append(word_text)
                word_duration = random.uniform(0.1, 0.8)
                word_end_time = current_time + word_duration
                
                abs_start = chunk_start_utc + datetime.timedelta(seconds=current_time)
                
                words_for_this_turn.append({
                    "word": word_text,
                    "start": round(current_time, 3),
                    "end": round(word_end_time, 3),
                    "probability": random.uniform(0.85, 0.99),
                    "speaker": speaker['name'],
                    "speaker_name": speaker['name'],
                    "absolute_start_utc": abs_start.isoformat(),
                    "matter_id": current_matter_id
                })
                current_time = word_end_time + random.uniform(0.05, 0.2)
            
            new_segment = {
                "start": round(turn_start_time, 3),
                "end": round(current_time, 3),
                "text": " ".join(turn_text_parts),
                "words": words_for_this_turn,
                "speaker": speaker['name']
            }
            all_segments.append(new_segment)
            
            last_matter_id = current_matter_id
        
        if matter_segments:
            matter_segments[-1]['end_time'] = current_time

        return all_segments, matter_segments

    def _create_speaker_flag(self, transcript, audio_filename, chunk_id, date_str):
        if not transcript: return {}
        flag_word = random.choice(transcript)

        speaker1, speaker2 = random.sample(self.speakers, k=2)

        top_score = random.uniform(0.75, 0.85)
        second_score = top_score - random.uniform(0.01, 0.04)

        return {
            "flag_id": f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}",
            "chunk_id": chunk_id,
            "timestamp_logged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_review",
            "source_file_name": audio_filename,
            "reason_for_flag": "Ambiguous speaker identification",
            "flag_type": "ambiguous_speaker",
            "candidates": [
                {
                    "name": speaker1['name'],
                    "score": round(top_score, 3),
                    "in_context": True
                },
                {
                    "name": speaker2['name'],
                    "score": round(second_score, 3),
                    "in_context": False
                }
            ],
            "text_preview": " ".join(w['word'] for w in transcript[5:15]),
            "segment_embedding": np.random.rand(self.embedding_dim).tolist()
        }

    def _create_matter_flag(self, transcript, audio_filename, chunk_id, date_str):
        if len(transcript) < 20 or len(self.matters) < 2:
            return None

        start_word = transcript[5]  
        end_word = transcript[19]
        text_snippet = " ".join(w['word'] for w in transcript[5:20])
        start_time = start_word['start']
        end_time = end_word['end']

        matter1, matter2 = random.sample(self.matters, 2)
        top_score = random.uniform(0.86, 0.95)
        second_score = top_score - random.uniform(0.005, 0.029)

        return {
            "flag_id": f"FLAG_{date_str.replace('-', '')}_{uuid.uuid4().hex[:12]}",
            "chunk_id": chunk_id,
            "timestamp_logged_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "status": "pending_review",
            "flag_type": "matter_conflict",  
            "source_file_name": audio_filename,
            "text_preview": text_snippet,
            "start_time": start_time,
            "end_time": end_time,
            "conflicting_matters": [
                {"matter_id": matter1['matter_id'], "name": matter1['name'], "score": round(top_score, 4)},
                {"matter_id": matter2['matter_id'], "name": matter2['name'], "score": round(second_score, 4)}
            ]
        }

    def _generate_master_log(self, all_words, day_start_utc, current_date):
        log_path = self.paths['database_folder'] / f"MASTER_DIALOGUE_{current_date.strftime('%Y-%m-%d')}.txt"
        
        assumed_tz_str = self.timings.get('assumed_recording_timezone', 'UTC')
        display_timezone = pytz.timezone(assumed_tz_str)
        master_log_timestamp_format = self.timings.get('master_log_timestamp_format', "%b%d, %Y - %H:%M")
        line_width = self.audio_suite_settings.get('master_log_line_width', 90)

        lines = []
        day_start_display = day_start_utc.astimezone(display_timezone).strftime(master_log_timestamp_format)
        lines.append(f"## Samson Master Log for {current_date.strftime('%Y-%m-%d')} ##\n")
        lines.append(f"Day recording started around: {day_start_display} ({assumed_tz_str})\n")

        current_speaker = None
        current_turn_text = ""

        for word in sorted(all_words, key=lambda x: x['start']):
            if word['speaker'] != current_speaker:
                if current_turn_text:
                    initial_indent = f"[{current_speaker}]: "
                    subsequent_indent = ' ' * len(initial_indent)
                    wrapped = textwrap.fill(current_turn_text, width=line_width, initial_indent=initial_indent, subsequent_indent=subsequent_indent)
                    lines.append(f"\n{wrapped}\n")
                current_speaker = word['speaker']
                current_turn_text = word['word']
            else:
                current_turn_text += " " + word['word']
        
        if current_turn_text:
            initial_indent = f"[{current_speaker}]: "
            subsequent_indent = ' ' * len(initial_indent)
            wrapped = textwrap.fill(current_turn_text, width=line_width, initial_indent=initial_indent, subsequent_indent=subsequent_indent)
            lines.append(f"\n{wrapped}\n")

        with open(log_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        logger.info(f"Generated master log: '{log_path.name}'.")

    def generate_task_data(self):
        """Generates a tasks.jsonl file with sample tasks."""
        logger.info("--- Generating Fake Task Data ---")
        
        tasks_file_path = Path(self.config.get('task_intelligence', {}).get('task_data_file', 'data/tasks/tasks.jsonl'))
        tasks_file_path.parent.mkdir(parents=True, exist_ok=True)
    
        tasks_to_create = []
        num_tasks = 10
    
        if not self.speakers or not self.generated_chunk_ids:
            logger.warning("No speakers or chunks generated, cannot create realistic tasks.")
            return
    
        owner = next((s for s in self.speakers if s['name'] == "System Administrator"), self.speakers[0])
    
        for i in range(num_tasks):
            status = random.choice(["pending_confirmation", "confirmed", "completed", "cancelled"])
            matter = random.choice(self.matters)
            source_chunk_id = random.choice(self.generated_chunk_ids)
    
            task = {
                "task_id": str(uuid.uuid4()),
                "owner_id": owner['name'],
                "status": status,
                "created_utc": (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=random.randint(1, 48))).isoformat(),
                "last_updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "title": f"Follow-up on {matter['name']} analysis",
                "description": f"This is a sample task generated for testing. It is related to matter '{matter['name']}' and originated from transcript chunk {source_chunk_id[:8]}...",
                "assignee_ids": [owner['name']],
                "matter_id": matter['matter_id'],
                "matter_name": matter['name'],
                "source_references": [
                    {
                        "source_type": "transcript",
                        "chunk_id": source_chunk_id
                    }
                ]
            }
            tasks_to_create.append(task)
        
        with open(tasks_file_path, 'w', encoding='utf-8') as f:
            for task in tasks_to_create:
                f.write(json.dumps(task) + '\n')
                
        logger.info(f"Generated {len(tasks_to_create)} tasks in '{tasks_file_path}'.")

def main():
    """Main function to run the data generator."""
    config = get_config()
    setup_logging(
        log_folder=config['paths']['log_folder'],
        log_file_name=config['paths']['log_file_name']
    )
    generator = TestDataGenerator(config)
    generator.confirm_and_clean_directories()
    generator.generate_speakers_and_matters()
    generator.generate_persistent_state_files()
    generator.generate_daily_data()
    generator.generate_task_data()

    print("\n--- ✅ Test Data Generation Complete! ---")
    print("You can now run the Samson Cockpit GUI to view the fake data.")
    print("  streamlit run gui.py")

if __name__ == "__main__":
    main()