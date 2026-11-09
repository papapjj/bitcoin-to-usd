# Dau bay lua chon va kiem dinh loi nhuan

## Pham vi

Bo nay dung than mang Five co dinh de hoc lai sau cach vao lenh cung mot lua chon dung ngoai:
HOLD, MARKET_BUY, MARKET_SELL, BUY_LIMIT, SELL_LIMIT, BUY_STOP, SELL_STOP.

Hop dong dau vao van la 137 gia tri, hai vai vao lenh dung chung mot quan sat tai moi thoi diem.
Nhan tuong lai chi dung lam muc tieu hoc va kiem dinh; khong duoc dua vao dau vao hay trang thai hoi tiep.

## Nhan va hinh hoc

Bo dung nhan dung nhip bid/ask, thoi gian cho khop, gia dat, gia khop, cat lo, chot loi, bien do thuan/bat loi va ket qua rong sau chenh lech gia.
Voi lenh cho, thoi gian giu 48 nen tinh tu luc khop va thoi gian het han khop la 5 nen, vi vay bien nhan qua khu dai nhat la 53 nen.
Mau so rui ro neo theo gia yeu cau luc dat lenh.

Hinh hoc hien co dinh o cat lo 1,5 ATR, chot loi 2R, do tre 0. Cau hinh yeu cau do lech 0,75 ATR duoc ma hoa vao o gan nhat; gia tri hieu dung cua o do la 0,588235 ATR.

Nhan nhan qua nhan hien rat thua: bo 16 tuan chi co 488 dong khac HOLD trong 322.360 dong, con kiem dinh va tap thu khong co dong khac HOLD.
Nhan chon theo ket qua tuong lai day hon chi duoc dung de hoc ngoai tuyen, khong dung de mo hoc tang cuong.

## Duong quyet dinh sau chi phi

Che do filled_value hoc loi nhuan trung binh tren lenh da khop. Xac suat khop chi la san loai bo lenh gan nhu khong the khop, khong duoc chia loi nhuan them lan nua.
Che do xac suat thang dung can hoa von cua hinh hoc 2R va co the tang trong so mau co loi trong nghien cuu.

Tep da danh dau hoc bay lua chon phai luu du cau hinh chay: che do quyet dinh, nguong, cach tinh diem, san xac suat khop, cac chi so hinh hoc va co bat mat gia tri.
SharedFiveTradingPolicy.require_current_architecture dung ngay khi thieu truong; cac trinh huan luyen cung kiem tra luu/nap lai. Vi vay tep cu thieu cau hinh khong duoc chay MT5.

## Trang thai danh muc

scripts/augment_seven_choice_state_audit.py ghep cac o vi the, tai khoan va thuc thi tu ban ghi MT5 cu theo dung thoi diem, dong thoi giu nguyen cac o thi truong hien tai.
Dong dang co vi the hoac lenh cho bi khoa chi con HOLD, vi hop dong bay lua chon chua co hanh dong quan tri vi the.

Lenh ghep phai doi chieu `feature_schema_hash` cua tep snapshot va tep state audit. Khi tao du lieu de hoc, truyen `--expected-feature-schema-hash <ma-day-du> --require-matching-feature-schema`; cong cu se dung loi neu mot nguon thieu ma hoac dung cau hinh khac. Neu bo qua cong kiem tra, ket qua van chi duoc danh dau nghien cuu va khong duoc coi la trang thai V193 dung hop dong.

Bo snapshot cu `BottomTop_Counterfactual_16w.set` dung ATR 14, EMA 20/60 va cach chia ATR cu. Bo nay chi de truy vet lich su. Khi xuat lai du lieu cho ung vien V193, dung `mt5_v2_runner/bottom_top_counterfactual_16w_v193_raw.ini` va `BottomTop_Counterfactual_16w_v193_raw.set`: ATR 30, EMA 50/200, cach chia ATR moi va ma day du `d5fd9...e3a402`. Neu ma trong tep khong trung ma trong cau hinh chay, phai dung truoc khi hoc.

src/self_growing_trader/causal_state_replay.py va scripts/augment_seven_choice_stateful_snapshots.py tao mot lich tong hop HOLD cong sau hanh dong de kiem tra kha nang doc trang thai.
Trang thai duoc ghi truoc khi xu ly nhip gia cua dong do, khong dung nhan tuong lai. Lich nay chi la nghien cuu, khong thay the viec ghi trang thai that tu MT5.

## Bang chung hien tai

Tren bon tuan, dau gia tri voi quan sat phang cho tap thu trung binh -0,321R, can duoi theo khoi ngay -0,401R va chi phat lenh mua.
Ghep trang thai cu lam ket qua thanh -0,247R, can duoi -0,305R, co ca mua va ban.
Xep hang theo xac suat thang voi trong so mau thang 1,0 con -0,123R, can duoi -0,155R.

Phep thu giu lai dung 3.502 thoi diem co trang thai roi chia theo thoi gian cho ket qua da ghep -0,458R, can duoi -0,708R; dau vao phang tren cung thoi diem la -0,217R, can duoi -0,422R.
Hoc 20 vong van -0,435R, can duoi -0,633R. Vi vay cach ghep tu ban ghi cu dang tao lech phan phoi va khong duoc dung de chay that.

Mot lan hoc tren toan bo 16 tuan cho trung binh +0,0046R, nhung toan bo trang thai ghep nam trong phan huan luyen; phan kiem dinh va tap thu khong co trang thai.
Con so duong nay khong chung minh mang doc duoc trang thai. Chi tiet nam trong outputs/seven_choice_teacher_causal_v16_20260917/v16_followup_diagnostic.json.

Da xuat lai 16 tuan bang cau hinh V193 dung. So nhan khac HOLD tang len 1.296 va xuat hien ca hai chieu. Dau gia tri theo loi nhuan da khop cho tap thu -0,196R moi lenh, can duoi theo khoi ngay -0,341R. Dau gia tri theo xac suat thang cho +0,0086R, nhung can duoi van -0,075R. Cau hinh da khop nhung bang chung kinh te van chua du de mo MT5 hay PPO. Bieu do moi o `outputs/seven_choice_teacher_causal_v17_20260917/v17_aligned_profit_comparison.png`.

Da thu them cach bo cac mau chua khop khoi sai so cua dau gia tri khi xep hang theo xac suat thang. Mot luot cho +0,031R moi lenh da khop tren tap thu, nhung can duoi theo khoi ngay -0,256R; phep lap lai cho ket qua am va kiem dinh xau hon. Vi vay dieu kien nay da duoc rut khoi cach hoc mac dinh; tep V18 chi giu de truy vet nghien cuu.

Phep do loc theo do rong chenh lech gia cho thay cac nguong rat thap co the lam trung binh tang, nhung tap kiem dinh chi con vai ngay va mau it. Day la dau hieu can xuat them nhieu khoang thoi gian, khong phai nguong duoc phep gan vao MT5. Chi tiet duoc luu tai `outputs/seven_choice_teacher_causal_v17_20260917/v17_spread_filter_probe.json`.

Bieu do doi chieu V17, V18 va loc chenh diem nam o `outputs/seven_choice_teacher_causal_v17_20260917/v17_followup_comparison.png`.

Da thu nhan “thang co dem” voi nguong loi nhuan rong 0,25R. Ket qua tap kiem dinh -0,101R va tap thu -0,034R, can duoi theo khoi ngay lan luot -0,165R va -0,173R. Nhan chat hon lam mat tin hieu, tep V19 da xoa phan trong so va chi giu bao cao.

Da hieu chinh diem xac suat rieng cho tung cach vao lenh bang phan hoc, sau do ap dung sang tap kiem dinh va tap thu. Cac nguong tot nhat tren tap kiem dinh van cho can duoi am va ket qua tap thu khong ben; khong giu cach nay.

Luot MT5 lon gan nhat co 379 vi the hoan tat, loi nhuan trung binh -0,2659R, can duoi -0,3353R. Cong tu hoc tang cuong van dong.

## Dieu kien buoc tiep theo

Can xuat lai nhan va trang thai trong cung mot luot MT5, dung dung cau hinh V193 tren ca phan huan luyen, kiem dinh va tap thu.
Ban ghi phai phu nhieu vi the, lenh cho va hanh dong quan tri, kem gia khop va vong doi dong. Sau do lap lai phep thu doi chung cung thoi gian, yeu cau can duoi theo khoi ngay duong truoc khi gan tep vao MT5 hoac bat PPO.

Bieu do so sanh nam o outputs/seven_choice_teacher_causal_v16_20260917/state_aware_profit_comparison.png.

## Cap nhat 2026-09-17: dong bo gia khop MT5

Phat lai MT5 cho thay lenh cho duoc khop tai dung `order_price`, trong khi bo
nhan cu lay gia bid/ask cham vao lenh. Chenh lech nay lam nhan cu lac quan hon
ket qua that. `SevenChoiceBuildConfig.pending_fill_price_mode` hien co hai che
do; du lieu moi dung `order`, con `touch` chi con de truy vet du lieu cu.

`seven_choice_runtime_contract` bay gio tu choi tap nhan thieu che do nay hoac
dung `touch` khi doi chieu MT5. Manifest moi ghi ro lenh cho khop dung gia dat,
va phep kiem tra tren 352.226 lan khop cho sai lech gia khop tru gia dat lon
nhat bang 0.

Sau khi khoa tac nhan quan ly ve `HOLD` va sua snapshot cham mot nhip trong
EA, phep chay MT5 doi chieu 2.500 goi van am: -152,17 USD, 21 giao dich, trung
binh -0,20095R moi vong doi. Hai dau gia tri V22 tren tap du lieu dong bo cung
am: -0,223R va -0,156R moi lenh da khop; can duoi theo khoi ngay lan luot
-0,268R va -0,238R. Ca hai chi la tep nghien cuu, khong duoc gan vao MT5 va
khong mo PPO.

Nhom nhan chon theo thong tin co truoc quyet dinh van gan nhu toan `HOLD`
(tap thu kin khong co dong khac `HOLD`), trong khi nhan chon theo ket qua
tuong lai dat khoang +0,40R. Day la dau hieu dau ra chua hoc du dieu kien
phan biet co hoi co loi, khong phai loi do cach tinh gia khop. Bieu do tong hop
nam o `outputs/diagnostics/v193_profit_diagnosis_20260917.png`, so lieu goc o
`outputs/diagnostics/v193_profit_diagnosis_20260917.json`.

Khong duoc dung ket qua am de ha cong kiem tra. Buoc tiep theo la tao them
nhan co dieu kien tren nhieu khoang thoi gian chua thay, huan luyen lai dau ra
voi than mang co dinh, roi lap lai MT5 theo cap cung ngay va yeu cau can duoi
loi nhuan sau chi phi duong truoc khi xem xet PPO.

Phep quet 80 to hop cat lo/chot loi tren cung sau tuan, voi gia khop dung gia
dat va chon hinh hoc bang lich su nhan qua, van cho tap thu -0,192R va can duoi
-0,266R. Do tuong quan cua 137 dac trung goc voi loi nhuan tung cach vao chi
nam trong khoang 0,02-0,18; bo do phi tuyen cung khong tao ra nhom 10 phan tram
co loi nhuan duong o tap thu. Vi vay can uu tien bo sung tin hieu nhan qua va
nhieu khoang thoi gian, thay vi tiep tuc doi nguong hay hinh hoc.

Bo do dieu kien don bien tim duoc 17 vung co can duoi duong tren tap kiem tra,
nhung khong co vung nao giu duoc can duoi duong khi chuyen sang tap thu kin.
Ba vung chi duong tren tap thu deu co can duoi am o tap kiem tra, nen duoc
giu rieng lam dau vet do sau, khong phai quy tac chay.



Da thu V20 voi dau bien do thuan va bat loi, trong so 0,25. Dau ra nay khong tao can duoi duong: tap thu trung binh +0,0156R nhung can duoi -0,235R.

Da chay mot khoang ngoai mau sach 16 tuan (24/04/2025-14/08/2025), cung ma dau vao V193 `d5fd9...e3a402`, 161.217 thoi diem. V20 voi nguong 0,3675 cho 8.214 lenh dong, trung binh +0,0008R va can duoi theo ngay -0,1141R. Them khoang cach diem 0,01 cho 6.664 lenh dong, trung binh +0,0163R nhung can duoi -0,1170R. Ket qua theo bon doan thoi gian lan luot -0,0099R, +0,0519R, -0,0034R, -0,0004R; dau hieu khong on dinh theo thoi gian.

Gioi han chenh lech gia rat thap 0,20 lam trung binh ngoai mau len +0,0768R va can duoi doc lap +0,0441R, nhung can duoi theo ngay van -0,0237R; day la phep mo ta sau khi da xem ngoai mau, khong duoc dung lam quy tac chay that. Hieu chinh theo V17 con lam ngoai mau am -0,0304R. Vi vay van khoa MT5 seed va PPO. Bieu do va so lieu nam o `outputs/seven_choice_teacher_causal_v17_20260917/v20_oos_diagnostic.png` va `v20_oos_filter_probe.json`.


Kiem tra kha nang xep hang tren ngoai mau cho thay AUC cua sau cach vao lenh chi 0,563-0,582, he so tuong quan thu bac voi loi nhuan chi 0,092-0,160. Nam 10 phan tram diem cao nhat cua cac lenh limit van gan hoa, con thi truong va lenh dung van am ro. Do do van de chinh la tin hieu dau vao chua tach duoc co hoi co loi sau chi phi; doi nguong don thuan se khong giai quyet duoc. So lieu nam o `outputs/seven_choice_teacher_causal_v17_20260917/v20_oos_ranking_diagnostic.json`.


Da thu diem loi nhuan sau chi phi `3*p_win - 1 - (2/3)*spread_feature_15`. Ngoai mau toan giai doan cho trung binh +0,0608R voi 4.470 lenh dong, nhung can duoi theo ngay -0,0326R. Tren V17, kiem dinh +0,0722R va tap thu +0,0626R deu co can duoi rat am (-0,504R va -0,382R), nen cong thuc nay chua ben va chi giu lam nhanh nghien cuu.


Phep thu hinh hoc ngoai mau voi cung diem V20 cho thay tang SL tu 1,5 len 2,0 roi 2,5 ATR lam loi nhuan trung binh cua bo loc khoang cach 0,01 tang tu +0,0163 len +0,0219 roi +0,0266R; can duoi theo ngay lan luot -0,1170, -0,0897 va -0,0575R. Day la dau hieu chi phi tren moi R dang qua lon, nhung van chua duong; can huan luyen lai dau ra tren tap phat trien cung hinh hoc truoc khi kiem tra MT5.

## Cap nhat 16 tuan voi gia khop dung gia dat

Da xuat lai khoang 24/04/2025-13/08/2025 tu MT5, gom 161.217 thoi diem va 322.434 dong vai. Tap moi dung gia khop lenh cho bang gia dat, thoi gian giu tinh tu luc khop, va giu mau so rui ro theo gia dat nhu hop dong MT5. Tep la `outputs/seven_choice_teacher_causal_v17_20260917/geometry_sl250_order_aligned_16w_after_v17.npz`; sai lech giua gia khop va gia dat bang 0 tren toan bo lan khop.

O cau hinh co dinh cat lo 2,5 ATR va chot loi 2R, loi nhuan gop trung binh cua tung cach vao lenh deu am: mua thi truong -0,057 den -0,082R, ban thi truong -0,039 den -0,082R, mua gioi han -0,041 den -0,066R, ban gioi han -0,065 den -0,079R, mua dung -0,013 den -0,030R, ban dung -0,032 den +0,005R tuy theo doan. Sau chi phi, tap thu kin lan luot -0,226, -0,208, -0,140, -0,176, -0,162 va -0,108R. Nghia la truoc ca chi phi, cach vao lenh chua co loi the binh quan.

Nhan nhan qua khu gan nhu chi con `HOLD`: 322.248/322.434 dong. Nhan chon theo ket qua tuong lai co 47% dong vao lenh va trung binh khoang +0,4R, nhung day la canh tren biet truoc tuong lai, chi duoc dung lam nhan hoc ngoai tuyen. Bo do cay tren bieu dien than mang van cho tap thu am; vi vay chua co bang chung de hoc lai dau ra ma giu than co dinh co the tao loi nhuan.

Quet 80 to hop cat lo/chot loi tren 16 tuan cho tap huan luyen -0,088R, tap kiem dinh +0,127R voi 58 lenh, tap thu kin -0,056R voi 47 lenh. Cau hinh duong o tap kiem dinh chi la ban dung voi cat lo 4 ATR va chot loi 0,25R; no khong lap lai duoc o tap thu, nen la qua khop. Do dai giu 6, 12, 24 va 48 nen cho lenh thi truong deu am ca loi nhuan gop lan sau chi phi.

Ket luan hien tai: thay doi cach dat lenh, nguong, hoac bat PPO chua giai quyet duoc loi nhuan trung binh. Can bo sung tin hieu nhan qua co tinh phan biet truoc thoi diem dat lenh, sau do hoc lai dau ra tren nhan khop dung gia dat va kiem tra lai MT5 theo cung khoang ngay. Bao cao va bieu do 16 tuan nam o `outputs/seven_choice_teacher_causal_v17_20260917/diagnostics_16w_order_aligned.json` va `diagnostics_16w_order_aligned.png`.

Bo sung phep thu loc theo chenh lech gia cho thay khi biet truoc cac dong co chenh lech duoi 0,5 ATR, can tren tuong lai co the len +0,82R; day chi la muc biet truoc ket qua. Khi dung cac dau ra hien co cua than mang de hoc chon huong trong cung vung nay, loi nhuan tap thu van am -0,106 den -0,267R moi lenh. Do dai giu 6, 12, 24 va 48 nen cho lenh thi truong cung khong tao loi nhuan gop duong. Bao cao chi tiet nam o `outputs/diagnostics/v23_low_spread_action_probe_16w.json`, `v23_market_horizon_probe_16w.json` va `v23_frozen_pattern_output_probe_16w.json`.

Dieu nay tach ro hai viec: chi phi chenh lech gia lam giam loi nhuan, nhung nguyen nhan lon hon la than mang chua phan biet duoc luc nao huong tuong lai se co loi. Giu nguyen than va chi hoc lai dau ra tren nhan hien tai se tiep tuc chon it lenh hoac chon sai; can them tin hieu co cau truc truoc quyet dinh (vi du bien dong, huong va khoang cach den moc gia theo chuoi thoi gian), roi moi tao ung vien moi.

Kiem tra them trang thai cuon cua LSTM va Transformer tren mau 6.144 dong (1.024 thoi diem moi doan) cho he so tuong quan voi loi nhuan tap thu chi 0,05-0,12; khi chon lenh theo diem nay, loi nhuan trung binh -0,125R va can duoi -0,171R. Day la phep mau nho de chan doan, chua thay the phat lai toan bo 16 tuan, nhung no cho thay chi them bo nho hien co cung chua tao duoc loi the ro rang.

Khi lay rieng 1%, 0,5% va 0,1% diem cao nhat cua bo du doan dau ra hien co, ca sau cach vao lenh deu van am tren tap thu; cach chon hanh dong tot nhat cung am -0,160R o 1% va -0,278R o 0,1%. Nghia la khong ton tai mot nhom diem cao hiem hoi co loi de cuu bang cach ha hoac nang nguong. So lieu nam o `outputs/diagnostics/v23_top_score_tail_16w.json`.

## Cap nhat 2026-09-17: tranh dem lap mau chong nhau

Mot mau nhan dai 48 nen khong phai 48 giao dich doc lap. Neu cap nhat lich su moi phut, cac ket qua dung chung cung doan gia tuong lai va lam can duoi qua lac quan. `SevenChoiceBuildConfig.action_history_stride_bars` da duoc them; dat bang 48 se chi cap nhat lich su mot lan trong moi khoi 48 nen. Che do mac dinh 1 duoc giu de tai tao du lieu cu, con bo moi phuc vu danh gia phai ghi ro gia tri nay trong ban do.

Bo 16 tuan voi cat lo 5 ATR, chot loi 0,25R, tre mot nhip va gia khop theo gia dat co trung binh duong khi xem tung phut, nhung can duoi theo ngay cua lenh da khop am nhe va phep chon khong chong nhau am. Khi loc mua dung trong vung dong cua nam trong khoang 0,870-0,948 cua bien 60 phut, ket qua phat lai khong chong nhau duong tren ca tap huan luyen, kiem dinh va tap thu; tuy nhien day van la quy tac tham do tren mot khoang 16 tuan, chua duoc MT5 xac nhan.

Vi vay chu “hinh hoc” chi ten khuon dat lenh (loai lenh, do tre, do lech, cat lo, chot loi, thoi gian het han). No khong phai la mot lop hoc hinh hoc va khong tu dong tao ra loi nhuan. Phep thu tiep theo phai dung cung mot quy tac loc, cung cach khong chong nhau, va doi chieu bang so du MT5 truoc khi gan vao tep ung vien hay bat PPO.

## Cap nhat 2026-09-17: kiem tra lai ty le 2R

Quet lai ca sau cach vao lenh voi cat lo 2,5 ATR va chot loi 2R, tre mot nhip,
gia khop dung gia dat va thoi gian giu 48 nen tu luc khop khong tim thay vung
dac trung nao co can duoi theo ngay duong o ca tap huan luyen, tap kiem dinh va
tap thu kin. Kieu tot nhat van am o it nhat mot doan; khong co quy tac de gan vao
MT5.

Khi lich su chi cap nhat moi 48 nen, tap nhan 2R co 0 lenh o tap huan luyen,
461 lenh mua dung o tap kiem dinh va 584 lenh mua dung o tap thu kin. Trung
binh tren cac lenh duoc chon lan luot la -0,033R va -0,118R; can duoi doc lap
lan luot -0,091R va -0,163R. Viec giam dem lap mau khong tao ra loi nhuan.

Mot loi thiet ke can giu ro: bo nhan nay chi cung hop dong khi ho so chay dat
`max_holding_bars=48`, che do dau vao thô, gia khop theo gia dat va cat lo/chot
loi 2R. Ho so ung vien thong thuong dang de `max_holding_bars=0` va chuan hoa
dau vao khac, khi do khong duoc phep dung bo nhan nay de ket luan ve MT5. Bao
cao gom o `outputs/diagnostics/v23_valid_rr_summary_20260917.json`.
Bieu do can duoi nam o `outputs/diagnostics/v23_valid_rr_diagnosis.png`.

Ket luan sau phep thu nay: loi nhuan am khong con do rieng cach tinh gia khop
hay do dem mau chong nhau; voi hop dong 2R hien tai, tin hieu truoc quyet dinh
chua tach duoc co hoi co loi. Chua hoc lai dau ra, chua tao checkpoint moi,
chua chay MT5 va chua bat PPO tu bo ket qua nay.

## Cap nhat 2026-09-17: thu cat lo rong hon de giam anh huong chi phi

Da giu nguyen chot loi 2R, tre mot nhip, thoi gian giu 48 nen, gia khop theo
gia dat va lich su chi cap nhat moi 48 nen. Hai bo du lieu duoc dat canh nhau:
cat lo 2,5 ATR voi lech thuc te 0,588 ATR, va cat lo 5 ATR voi cung lech.

Cat lo 5 ATR lam chi phi trung binh tren moi lenh giam gan mot nua. O tap thu,
loi nhuan sau chi phi theo sau cach vao lenh lan luot la -0,095, -0,098,
-0,116, -0,126, -0,080 va -0,074R; bo cat lo 2,5 ATR cung thu tu la
-0,226, -0,208, -0,283, -0,267, -0,190 va -0,162R. Nghia la cat lo rong hon
la huong tot hon trong ba bo da thu, nhung ca sau cach van am va phan loi
nhuan gop cua lenh van chua ben.

Da thu them lech dat 3 ATR voi cat lo 5 ATR. Ty le khop cua lenh cho giam rat
manh (chi con khoang 5-13 phan tram tuy cach), trong khi loi nhuan tap thu cua
lenh cho nam trong khoang -0,072 den -0,160R. Bo nay bi loai bo vi khop it hon
ma khong co loi nhuan tot hon.

Phep thu nay chi tao ra bo nhan va bao cao kinh te, chua tao checkpoint moi,
chua gan vao MT5 va chua bat PPO. Muon dung cat lo 5 ATR phai huan luyen lai
dau ra voi dung hop dong cat lo 5 ATR, chot loi 10 ATR, roi chay MT5 doi chung
tren cung khoang ngay. Chi khi can duoi loi nhuan sau chi phi duong o ca ba
doan huan luyen, kiem dinh va tap thu moi duoc xem xet gan ung vien.

So lieu va bieu do: `outputs/diagnostics/v23_geometry_economic_comparison_20260917.json`,
`outputs/diagnostics/v23_geometry_economic_comparison.png` va
`outputs/seven_choice_teacher_causal_v17_20260917/geometry_sl500_rr2_delay0_stride48_order_aligned_16w_after_v17.npz`.

## Pham vi dau ra da hoc

Ba bo nhan `h12`, `h24` va `h48` deu luu day du 16 truong ket qua cho moi
cach vao: khop, gia dat, gia khop, thoi gian cho theo nen va theo nhip, do tre,
do lech, cat lo, chot loi, bien do thuan/bat loi, loi gop, chi phi, loi rong,
so nen thoat va ma thoat. Bao cao doi chieu nam o
`outputs/diagnostics/v23_output_coverage_audit_20260918.json`.

Dau hanh dong dang hoc kha nang khop va loi rong; dau bien do thuan/bat loi chi
duoc hoc khi bat `path_excursion_loss_weight`. Dau gia dat, gia khop, thoi gian
cho den khi khop, loi gop, chi phi rieng va ly do thoat hien chi la nhan thong
ke; chung chua co dau phu rieng. Cac chi so cat lo/chot loi co dau phan loai,
nhung ung vien hinh hoc co dinh ghi de bang chi so cua hop dong. Khong mo rong
cac dau nay vao duong chay hoac bat PPO khi chua co phat lai cung dieu kien voi
can duoi loi nhuan duong.

## Cap nhat 2026-09-17: thoi gian giu 24 nen

Voi cung cat lo 5 ATR, chot loi 2R, lech 0,588 ATR va gia khop dung gia dat,
giam thoi gian giu tu 48 xuong 24 nen lam ket qua tap thu tot hon nhe. Sau
chi phi, sau cach vao lenh lan luot o 24 nen la -0,087, -0,090, -0,107,
-0,114, -0,074 va -0,067R; o 48 nen la -0,095, -0,098, -0,116, -0,126,
-0,080 va -0,074R. Phan lon lenh van ket thuc do het thoi gian, nen day la
giam muc am chu chua phai loi the da xac nhan.

Thu cat lo 4 ATR va giu 24 nen lai kem hon cat lo 5 ATR o ca sau cach. Vi vay
ung vien tiep theo neu duoc tao phai uu tien cat lo 5 ATR, chot loi 10 ATR va
giu 24 nen. Ho so nay da co tai `configs/experiments/edge_discovery_v193_value_oos_fixed_rr2_sl500_h24.yaml`,
nhung chua co tệp trong so nao phu hop va chua duoc chay MT5.

Kiem tra bao ve sau chuoi thua tren 178 giao dich da ghi cho thay quy tac
chan sau 3 lenh thua lien tiep co the giam tong lo uoc tinh tu -47,35R xuong
-6,11R, nhung no bo qua 78% yeu cau va lam xau 3 trong 7 sổ lệnh. Day chi la
mo phong tu so lieu cu, khong phai bang chung nhan qua; khong bat mac dinh
cho den khi co mot phep chay MT5 doi chung co cung ngay va cung tệp trong so.

So lieu bao ve nam o `outputs/diagnostics/v23_entry_loss_guard_event_aligned_20260917.json`.
Bieu do so sanh thoi gian giu nam o `outputs/diagnostics/v23_geometry_horizon_comparison.png`.

## Cap nhat 2026-09-18: thu them thoi gian giu 12 nen

Giu nguyen cat lo 5 ATR, chot loi 2R, lech dat thuc te 0,588 ATR, tre mot
nhip, gia khop theo gia dat va lich su khong chong nhau; chi rut thoi gian giu
tu 24 xuong 12 nen. Tap thu cho sau cach vao lenh lan luot -0,085, -0,087,
-0,098, -0,107, -0,073 va -0,067R. So voi 24 nen, muc am giam khoang
0,002-0,007R moi lenh; day la ket qua tot nhat trong ba moc 12, 24 va 48 nen,
nhung can duoi theo ngay van lan luot -0,093, -0,097, -0,108, -0,120,
-0,083 va -0,077R, deu am.

Ty le het thoi gian van rat cao (gan 0,985-0,996 tuy cach), nen viec rut ngan
khong tao ra nhieu lenh dong bang muc chot loi. Cac lenh mua dung va ban dung
van it am nhat, nhung chua co can duoi duong de gan vao MT5. Ho so khop hop
dong nay da co tai `configs/experiments/edge_discovery_v193_value_oos_fixed_rr2_sl500_h12.yaml`;
chua tao trong so moi, chua chay MT5 va chua bat PPO.

So lieu day du nam o `outputs/diagnostics/v23_geometry_horizon_comparison_20260918.json`;
bieu do nam o `outputs/diagnostics/v23_geometry_horizon_comparison_20260918.png`.

## Cap nhat 2026-09-18: hieu chinh tu trang thai an theo thoi gian

Kiem tra duong chay cho thay mat gia tri sau chi phi truoc day chi nhin nut
goc tu dau vao hien tai. Trang thai an sau LSTM va bo bien doi co khac biet
theo vai mua/ban va lich su, nhung he so noi vao mat gia tri mac dinh bang 0;
vi vay thong tin nay khong the doi thu hang bay lua chon.

Da them mot nhanh hieu chinh sau, gom 6 gia tri tu trang thai an, khoi tao
bang 0 va gioi han bien do. Nhanh nay khong sua than mang, LSTM, bo bien doi,
137 dau vao hay 20 dau ra; mac dinh tat va chi duoc bat bang
`--hidden-value-residual-enabled` trong ho so nghien cuu co
`--allow-research-only`. Neu duoc bat, nhanh hoc net-R sau phi tren cac lua
chon hop le cua hai chu so vao lenh, sau do duoc cong vao mat gia tri dung de
chon lenh. Chua duoc xem la bang chung loi nhuan va khong duoc tu dong bat
PPO.

Phep thu ky thuat da them hai bao ve: checkpoint cu van cho ket qua y het khi
nhanh nay tat; khi bat ma thieu trang thai an thi phat loi thay vi am tham
bo qua. Bao ve nay giup tranh tinh huong so lieu danh gia khac duong chay
MT5.

## Cap nhat 2026-09-18: doi chung quy tac loc theo ba moc thoi gian

Quy tac duoc tim thay o moc 48 nen la mua cho khi `m15_ema_slope_atr` nam
trong khoang `[-0,31898; -0,11018)`. Khi khoa nguyen quy tac roi ap vao tap
thu cua ba moc, trung binh lenh da khop sau phi la -0,018R o 12 nen, +0,006R
o 24 nen va +0,040R o 48 nen. Can duoi theo ngay lan luot la -0,024R,
+0,015R va +0,090R.

Tach rieng tap 48 nen thanh hai nua lien tiep cho thay nua dau chi dat
0,016R moi lenh da khop, can duoi theo ngay +0,001R; nua sau dat 0,068R va
+0,106R. Ket qua nay cho thay co the co thay doi trang thai thi truong theo
thoi gian, khong du de ket luan quy tac ben vung. Quy tac van bi giu o muc
nghien cuu vi trung binh tren tap huan luyen va tap kiem dinh van am, moc 12
nen con am, va nua dau moc 48 nen chua tao bien an toan ro rang.

Bao cao doi chung day du o
`outputs/diagnostics/v23_fixed_rule_cross_horizon_confirmation_20260918.json`.

Quet tiep 10 dac trung thi truong, bien dong va thoi gian theo tung cap voi
8 khoang phan vi, yeu cau toi thieu 256 mau moi doan. Khong co cap nao vuot
dong thoi trung binh va can duoi theo ngay khong am o ca tap huan luyen va tap
kiem dinh tai 12, 24 hay 48 nen. Dieu nay bac bo viec them mot bo loc don
gian tu hai dac trung se tu dong tao loi nhuan tren tap hien tai.

Bao cao tai `outputs/diagnostics/v23_pair_causal_market_filters_strict_h12_20260918.json`,
`outputs/diagnostics/v23_pair_causal_market_filters_strict_h24_20260918.json` va
`outputs/diagnostics/v23_pair_causal_market_filters_strict_h48_20260918.json`.
