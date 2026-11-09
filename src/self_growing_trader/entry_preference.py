"""Delayed causal HOLD-versus-entry labels for Direct-RNN Full-RL."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping
import numpy as np
ENTRY_PREFERENCE_ACTIONS=(0,1,2,3,4,5,6)
HOLD_ACTION,MARKET_BUY,MARKET_SELL,BUY_LIMIT,SELL_LIMIT,BUY_STOP,SELL_STOP=ENTRY_PREFERENCE_ACTIONS
HIERARCHICAL_TRADE_TARGET_MODE="hierarchical_on_policy_trade_v2"
COUNTERFACTUAL_CONSENSUS_TARGET_MODE="counterfactual_consensus_v3"
TIMING_FIRST_TARGET_MODE="relative_entry_timing_v1"
@dataclass(frozen=True)
class EntryPathBar:
    high:float; low:float; close:float; spread:float
@dataclass(frozen=True)
class EntryPreferenceRecord:
    trace_id:str; sequence:int; decision_time:str; selected_action:int; bid:float; ask:float; atr:float; sl_atr:float; rr:float
    entry_prices:tuple[float,float,float,float,float,float]; bars:tuple[EntryPathBar,...]
    sample_scope:str="flat_v1"; positions_before:int=0; orders_before:int=0
    # V191 keeps the short immutable M1 path for counterfactual execution, but
    # stores sparse long-horizon observations separately so H4/H6/H10 labels
    # never alias to the last available short-path close.
    future_closes:tuple[float,...]=()
    future_volatilities:tuple[float,...]=()
    observed_bars:int=0
@dataclass(frozen=True)
class EntryUtilityComponents:
    action:int; filled:bool; fill_bar:int; terminal_r:float; mfe_r:float; mae_r:float; bars_to_positive:int; observed_bars:int
    direction_r:float; timing_r:float; geometry_r:float; execution_cost_r:float; quality_r:float
    def as_dict(self): return {name:getattr(self,name) for name in self.__dataclass_fields__}
@dataclass(frozen=True)
class EntryPathDiagnostics:
    """Thong tin thu tu va thoi gian tren path sau quyet dinh.

    ``outcome_class``: 0=khong khop, 1=chua cham SL/TP, 2=SL truoc,
    3=TP truoc. Neu mot bar cham ca hai, quy tac bao thu SL-truoc duoc dung
    vi du lieu OHLC M1 khong cho biet thu tu tick trong bar.
    """
    action:int; filled:bool; fill_bar:int; outcome_class:int
    time_to_mfe:float; time_to_mae:float; underwater_fraction:float
    recovery_fraction:float; positive_persistence:float
    def as_dict(self): return {name:getattr(self,name) for name in self.__dataclass_fields__}
@dataclass(frozen=True)
class EntryTimingComparison:
    """Action-local comparison of entering now versus later on one causal path."""
    action:int; current_utility_r:float; best_utility_r:float; best_delay_bars:int
    second_best_utility_r:float; delay_margin_r:float
    timing_advantage_r:float; enter_now_probability:float; valid_delays:int
    def as_dict(self): return {name:getattr(self,name) for name in self.__dataclass_fields__}
def _number(value,default=0.0):
    try: parsed=float(value)
    except (TypeError,ValueError): return default
    return parsed if np.isfinite(parsed) else default
def _integer(value,default=0):
    try:return int(float(value))
    except (TypeError,ValueError):return int(default)
def _parse_prices(raw):
    values=tuple(_number(x,float("nan")) for x in str(raw or "").split("|"))
    return values if len(values)==6 and all(np.isfinite(x) and x>0 for x in values) else None
def _parse_path(raw):
    bars=[]
    for fragment in str(raw or "").split(";"):
        values=tuple(_number(x,float("nan")) for x in fragment.split("|"))
        if len(values)!=4 or not all(np.isfinite(x) for x in values):continue
        high,low,close,spread=values
        if high>0 and low>0 and close>0 and high>=low and spread>=0:bars.append(EntryPathBar(high,low,close,spread))
    return tuple(bars)
def _parse_finite_pipe(raw):
    values=tuple(_number(x,float("nan")) for x in str(raw or "").split("|") if str(x).strip()!="")
    return values if values and all(np.isfinite(x) for x in values) else ()
def parse_entry_preference_row(row:Mapping[str,object]):
    trace_id=str(row.get("trace_id","") or "").strip(); prices=_parse_prices(row.get("entry_prices_pipe")); bars=_parse_path(row.get("path_pipe"))
    atr,sl_atr,rr=(_number(row.get(n),float("nan")) for n in ("atr","sl_atr","rr"))
    if not trace_id or prices is None or not bars or not all(np.isfinite(v) and v>0 for v in (atr,sl_atr,rr)):return None
    try: sequence=int(float(row.get("sequence",0) or 0)); selected=int(float(row.get("selected_action",0) or 0))
    except (TypeError,ValueError):return None
    scope=str(row.get("sample_scope","flat_v1") or "flat_v1").strip().lower()
    if scope not in {"flat_v1","entry_marginal_v2"}:return None
    return EntryPreferenceRecord(
        trace_id,sequence,str(row.get("decision_time","") or ""),selected,
        _number(row.get("bid")),_number(row.get("ask")),atr,sl_atr,rr,prices,bars,
        scope,max(0,_integer(row.get("positions_before"))),max(0,_integer(row.get("orders_before"))),
        _parse_finite_pipe(row.get("future_close_pipe")),
        _parse_finite_pipe(row.get("future_volatility_pipe")),
        max(len(bars),_integer(row.get("observed_bars"),len(bars))),
    )
def _entry_price(record,action):
    index={1:0,2:1,3:2,4:3,5:4,6:5}.get(int(action)); return None if index is None else float(record.entry_prices[index])
def _side(action):return 1 if int(action) in {1,3,5} else -1
def _pending_filled(action,entry,bar):
    return {3:bar.low+bar.spread<=entry,4:bar.high>=entry,5:bar.high+bar.spread>=entry,6:bar.low<=entry}.get(action,action in {1,2})
def replay_entry_components(record,action):
    action=int(action)
    if action==0:return EntryUtilityComponents(0,False,-1,0.,0.,0.,-1,len(record.bars),0.,0.,0.,0.,0.)
    entry=_entry_price(record,action); risk=float(record.sl_atr*record.atr)
    if entry is None or risk<=1e-12:return EntryUtilityComponents(action,False,-1,0.,0.,0.,-1,len(record.bars),0.,0.,0.,0.,0.)
    side=_side(action); sl=entry-side*risk; tp=entry+side*risk*record.rr; filled=action in {1,2}; fill_bar=0 if filled else -1
    mfe=mae=terminal=0.; first=-1; observed=0
    for index,bar in enumerate(record.bars):
        if not filled:
            if not _pending_filled(action,entry,bar):continue
            filled=True; fill_bar=index
        observed+=1
        if side>0: stop,target=bar.low<=sl,bar.high>=tp; favorable,adverse=(bar.high-entry)/risk,(bar.low-entry)/risk
        else: stop,target=bar.high+bar.spread>=sl,bar.low+bar.spread<=tp; favorable,adverse=(entry-bar.low-bar.spread)/risk,(entry-bar.high-bar.spread)/risk
        mfe=max(mfe,favorable); mae=min(mae,adverse)
        if first<0 and favorable>.05:first=observed
        if stop:terminal=-1.;break
        if target:terminal=float(record.rr);break
    if not filled:return EntryUtilityComponents(action,False,-1,0.,0.,0.,-1,0,0.,0.,0.,0.,0.)
    if terminal==0:
        final=record.bars[-1]; mark=final.close if side>0 else final.close+final.spread; terminal=float(np.clip(side*(mark-entry)/risk,-3.,5.))
    direction=float(np.clip(.55*terminal+.45*np.clip(mfe,0.,3.),-3.,5.)); horizon=max(1,len(record.bars)); delay=1. if first<0 else (first-1)/horizon
    timing=float(np.clip(.65*mae-.20*delay-.15*max(0,fill_bar)/horizon,-2.,0.)); geometry=float(np.clip(terminal-direction,-2.,2.)); cost=float(-np.clip(max(0.,record.ask-record.bid)/risk,0.,1.))
    quality=float(np.clip(.55*direction+.30*timing+.10*geometry+.05*cost,-3.,5.))
    return EntryUtilityComponents(action,True,fill_bar,terminal,mfe,mae,first,observed,direction,timing,geometry,cost,quality)
def replay_entry_path_diagnostics(record,action):
    """Tai tao thong tin path ma MAE/MFE don le khong the hien duoc."""
    action=int(action)
    if action==0:
        return EntryPathDiagnostics(0,False,-1,0,0.,0.,0.,0.,0.)
    entry=_entry_price(record,action); risk=float(record.sl_atr*record.atr)
    if entry is None or risk<=1e-12:
        return EntryPathDiagnostics(action,False,-1,0,0.,0.,0.,0.,0.)
    side=_side(action); sl=entry-side*risk; tp=entry+side*risk*record.rr
    filled=action in {1,2}; fill_bar=0 if filled else -1
    favorable_values=[]; adverse_values=[]; close_values=[]; outcome=1
    for index,bar in enumerate(record.bars):
        if not filled:
            if not _pending_filled(action,entry,bar):continue
            filled=True; fill_bar=index
        if side>0:
            stop,target=bar.low<=sl,bar.high>=tp
            favorable=(bar.high-entry)/risk; adverse=(bar.low-entry)/risk
            close_r=(bar.close-entry)/risk
        else:
            stop,target=bar.high+bar.spread>=sl,bar.low+bar.spread<=tp
            favorable=(entry-bar.low-bar.spread)/risk
            adverse=(entry-bar.high-bar.spread)/risk
            close_r=(entry-bar.close-bar.spread)/risk
        favorable_values.append(float(favorable)); adverse_values.append(float(adverse))
        close_values.append(float(close_r))
        if stop:
            outcome=2; break
        if target:
            outcome=3; break
    if not filled or not close_values:
        return EntryPathDiagnostics(action,False,-1,0,0.,0.,0.,0.,0.)
    count=len(close_values); mfe_index=int(np.argmax(favorable_values)); mae_index=int(np.argmin(adverse_values))
    underwater=float(np.mean(np.asarray(close_values,dtype=float)<0.0))
    mae=float(adverse_values[mae_index]); terminal=float(close_values[-1])
    span=max(1e-6,float(max(favorable_values))-mae)
    recovery=float(np.clip((terminal-mae)/span,0.,1.))
    positive_indices=np.flatnonzero(np.asarray(close_values,dtype=float)>0.0)
    if positive_indices.size:
        first_positive=int(positive_indices[0])
        persistence=float(np.mean(np.asarray(close_values[first_positive:],dtype=float)>0.0))
    else:persistence=0.0
    denominator=max(1,count-1)
    return EntryPathDiagnostics(
        action,True,fill_bar,outcome,
        float(mfe_index/denominator),float(mae_index/denominator),underwater,
        recovery,persistence,
    )
def replay_entry_path_returns(record,action):
    """Tra ve chuoi loi suat R sau khi mot entry cu the da khop.

    Chuoi nay chi duoc tao tu cac bar da dong sau quyet dinh. Lenh cho chua
    khop khong tao gia tri; neu cung mot bar cham ca SL va TP thi dung quy tac
    bao thu SL-truoc giong ``replay_entry_components``.
    """
    action=int(action)
    if action==0:return ()
    entry=_entry_price(record,action); risk=float(record.sl_atr*record.atr)
    if entry is None or risk<=1e-12:return ()
    side=_side(action); sl=entry-side*risk; tp=entry+side*risk*record.rr
    filled=action in {1,2}; values=[]
    for bar in record.bars:
        if not filled:
            if not _pending_filled(action,entry,bar):continue
            filled=True
        if side>0:
            stop,target=bar.low<=sl,bar.high>=tp
            close_r=(bar.close-entry)/risk
        else:
            stop,target=bar.high+bar.spread>=sl,bar.low+bar.spread<=tp
            close_r=(entry-bar.close-bar.spread)/risk
        if stop:values.append(-1.0);break
        if target:values.append(float(record.rr));break
        values.append(float(np.clip(close_r,-3.,5.)))
    return tuple(values)
def replay_entry_utility(record,action,*,mode="terminal_r_v1"):
    item=replay_entry_components(record,action); mode=str(mode or "terminal_r_v1").lower()
    if mode=="terminal_r_v1":return float(item.terminal_r)
    if mode=="path_quality_v1":return float(item.quality_r)
    raise ValueError("entry utility mode must be terminal_r_v1 or path_quality_v1")
def replay_entry_utilities(record,*,mode="terminal_r_v1"):
    return np.asarray([replay_entry_utility(record,a,mode=mode) for a in ENTRY_PREFERENCE_ACTIONS],dtype=float)
def replay_entry_component_matrix(record):return tuple(replay_entry_components(record,a) for a in ENTRY_PREFERENCE_ACTIONS)
def replay_entry_path_diagnostic_matrix(record):return tuple(replay_entry_path_diagnostics(record,a) for a in ENTRY_PREFERENCE_ACTIONS)


def _record_delayed(record:EntryPreferenceRecord,delay_bars:int):
    """Re-anchor the same six entry styles after ``delay_bars`` closed M1 bars.

    The quote at the delayed decision is the close of the last fully observed
    bar.  Simulation starts at the following bar, so the high/low that formed
    the delayed quote can never leak into the delayed outcome.
    """
    delay=max(0,int(delay_bars))
    if delay<=0:return record
    if delay>=len(record.bars):return None
    quote_bar=record.bars[delay-1]
    bid=float(quote_bar.close); ask=bid+max(0.,float(quote_bar.spread))
    old=tuple(float(value) for value in record.entry_prices)
    prices=(
        ask,
        bid,
        ask+(old[2]-float(record.ask)),
        bid+(old[3]-float(record.bid)),
        ask+(old[4]-float(record.ask)),
        bid+(old[5]-float(record.bid)),
    )
    return EntryPreferenceRecord(
        trace_id=record.trace_id,sequence=record.sequence,
        decision_time=record.decision_time,selected_action=record.selected_action,
        bid=bid,ask=ask,atr=record.atr,sl_atr=record.sl_atr,rr=record.rr,
        entry_prices=prices,bars=tuple(record.bars[delay:]),
        sample_scope=record.sample_scope,positions_before=record.positions_before,
        orders_before=record.orders_before,observed_bars=max(0,record.observed_bars-delay),
    )


def replay_entry_timing_comparisons(
    record:EntryPreferenceRecord,
    *,
    delay_bars:Iterable[int]=(1,2,4,8,12),
    utility_mode:str="path_quality_v1",
    temperature_r:float=.25,
):
    """Compare each action only with delayed versions of that same action.

    This deliberately does not compare BUY with SELL or MARKET with pending
    entries.  Direction/style learning therefore cannot masquerade as timing.
    """
    delays=tuple(sorted({int(value) for value in delay_bars if int(value)>0}))
    delayed_records=tuple(
        (delay,_record_delayed(record,delay)) for delay in delays
        if delay<len(record.bars)
    )
    comparisons=[]
    for action in ENTRY_PREFERENCE_ACTIONS[1:]:
        current=float(replay_entry_utility(record,action,mode=utility_mode))
        alternatives=[(0,current)]
        for delay,delayed in delayed_records:
            if delayed is not None and delayed.bars:
                alternatives.append((delay,float(replay_entry_utility(delayed,action,mode=utility_mode))))
        ranked_alternatives=sorted(alternatives,key=lambda item:(item[1],-item[0]),reverse=True)
        best_delay,best=ranked_alternatives[0]
        second_best=(ranked_alternatives[1][1] if len(ranked_alternatives)>1 else best)
        delayed_best=max((value for delay,value in alternatives if delay>0),default=current)
        advantage=float(np.clip(current-delayed_best,-3.,3.))
        probability=1./(1.+float(np.exp(-np.clip(advantage/max(.05,float(temperature_r)),-60.,60.))))
        comparisons.append(EntryTimingComparison(
            action=action,current_utility_r=current,best_utility_r=float(best),
            best_delay_bars=int(best_delay),second_best_utility_r=float(second_best),
            delay_margin_r=float(max(0.,best-second_best)),timing_advantage_r=advantage,
            enter_now_probability=float(probability),valid_delays=max(0,len(alternatives)-1),
        ))
    return tuple(comparisons)


def timing_gate_target(
    comparisons:Iterable[EntryTimingComparison],
    *,
    allowed_actions=ENTRY_PREFERENCE_ACTIONS,
    timing_margin_r:float=.05,
    trade_margin_r:float=.10,
    temperature_r:float=.25,
    consensus_actions:int=2,
):
    """Return a behavior-independent ENTER-NOW probability.

    An action family supports ENTER-NOW only when its current utility is above
    HOLD and it is not improved by waiting.  The top two viable families are
    averaged to avoid turning one lucky hindsight path into a trade label.
    """
    allowed={int(action) for action in allowed_actions if 1<=int(action)<=6}
    rows=[row for row in comparisons if row.action in allowed and row.valid_delays>0]
    if not rows:return 0.,{"eligible":False,"reason":"no_delayed_path"}
    ranked=sorted(rows,key=lambda row:max(row.current_utility_r,row.best_utility_r),reverse=True)
    width=max(1,min(int(consensus_actions),len(ranked)))
    selected=ranked[:width]
    current=float(np.mean([row.current_utility_r for row in selected]))
    advantage=float(np.mean([row.timing_advantage_r for row in selected]))
    scale=max(.05,float(temperature_r))
    current_probability=1./(1.+float(np.exp(-np.clip((current-float(trade_margin_r))/scale,-60.,60.))))
    timing_probability=1./(1.+float(np.exp(-np.clip((advantage-float(timing_margin_r))/scale,-60.,60.))))
    target=float(np.clip(current_probability*timing_probability,0.,1.))
    return target,{
        "eligible":True,"reason":"ok","current_utility_r":current,
        "timing_advantage_r":advantage,"consensus_actions":width,
        "selected_actions":[int(row.action) for row in selected],
    }
def preference_target(utilities:Iterable[float],*,allowed_actions=ENTRY_PREFERENCE_ACTIONS,temperature=.50):
    values=np.asarray(tuple(utilities),dtype=float).reshape(-1)
    if values.size!=7:raise ValueError("entry preference utilities must contain HOLD plus six entry actions")
    mask=np.zeros(7,dtype=bool)
    for action in allowed_actions:
        if int(action) in ENTRY_PREFERENCE_ACTIONS:mask[int(action)]=True
    mask[0]=True; safe=np.clip(np.where(np.isfinite(values),values,0.),-3.,5.); masked=np.where(mask,safe/max(.05,float(temperature)),-np.inf); shifted=masked-np.max(masked[mask]); weights=np.where(mask,np.exp(np.clip(shifted,-60.,60.)),0.)
    return weights/max(float(weights.sum()),1e-12),mask


def hierarchical_preference_target(
    utilities: Iterable[float],
    *,
    allowed_actions=ENTRY_PREFERENCE_ACTIONS,
    selected_action: int | None = None,
    action_temperature: float = .50,
    trade_margin_r: float = .10,
    trade_temperature: float = .25,
):
    """Build an honest P(TRADE) plus P(action | TRADE) target.

    A settled trade is judged by the action the policy actually selected.  It
    is never relabelled as TRADE merely because one of five unselected,
    hindsight alternatives happened to win.  A HOLD sample has no executed
    entry, so its group target uses the uniform mean over legal entries rather
    than an optimistic max.  The complete utility vector is still available
    to rank actions conditionally when the group target supports trading.
    """
    values = np.asarray(tuple(utilities), dtype=float).reshape(-1)
    if values.size != 7:
        raise ValueError("entry preference utilities must contain HOLD plus six entry actions")
    mask = np.zeros(7, dtype=bool)
    for action in allowed_actions:
        if int(action) in ENTRY_PREFERENCE_ACTIONS:
            mask[int(action)] = True
    mask[HOLD_ACTION] = True
    entry_mask = mask.copy()
    entry_mask[HOLD_ACTION] = False
    target = np.zeros(7, dtype=float)
    if not np.any(entry_mask):
        target[HOLD_ACTION] = 1.0
        return target, mask
    safe = np.clip(np.where(np.isfinite(values), values, 0.0), -3.0, 5.0)
    try:
        selected = int(selected_action) if selected_action is not None else HOLD_ACTION
    except (TypeError, ValueError):
        selected = HOLD_ACTION
    if selected != HOLD_ACTION and 0 <= selected < safe.size and bool(entry_mask[selected]):
        trade_utility = float(safe[selected])
    else:
        trade_utility = float(np.mean(safe[entry_mask]))
    relative = (
        trade_utility - float(safe[HOLD_ACTION]) - float(trade_margin_r)
    ) / max(.05, float(trade_temperature))
    trade_probability = 1.0 / (1.0 + float(np.exp(-np.clip(relative, -60.0, 60.0))))
    entry_logits = np.where(
        entry_mask,
        safe / max(.05, float(action_temperature)),
        -np.inf,
    )
    shifted = entry_logits - float(np.max(entry_logits[entry_mask]))
    conditional = np.where(entry_mask, np.exp(np.clip(shifted, -60.0, 60.0)), 0.0)
    conditional /= max(float(np.sum(conditional)), 1e-12)
    target[HOLD_ACTION] = 1.0 - trade_probability
    target[entry_mask] = trade_probability * conditional[entry_mask]
    return target, mask


def counterfactual_consensus_preference_target(
    utilities: Iterable[float],
    *,
    allowed_actions=ENTRY_PREFERENCE_ACTIONS,
    action_temperature: float = .50,
    trade_margin_r: float = .10,
    trade_temperature: float = .25,
    consensus_actions: int = 2,
):
    """Build a behavior-independent delayed target from all legal actions.

    The old V2 group target reused the action selected by the behavior policy
    many bars earlier.  Once the label matured, that made a stale policy
    choice part of the target itself.  V3 separates the two questions:

    * rank every legal entry from the immutable counterfactual path;
    * label TRADE from the mean of the best ``consensus_actions`` utilities.

    Requiring more than one supporting entry style prevents a single lucky
    hindsight action from turning HOLD into TRADE, while the target no longer
    depends on which action the old policy happened to execute.
    """
    values = np.asarray(tuple(utilities), dtype=float).reshape(-1)
    if values.size != 7:
        raise ValueError("entry preference utilities must contain HOLD plus six entry actions")
    mask = np.zeros(7, dtype=bool)
    for action in allowed_actions:
        if int(action) in ENTRY_PREFERENCE_ACTIONS:
            mask[int(action)] = True
    mask[HOLD_ACTION] = True
    entry_mask = mask.copy()
    entry_mask[HOLD_ACTION] = False
    target = np.zeros(7, dtype=float)
    if not np.any(entry_mask):
        target[HOLD_ACTION] = 1.0
        return target, mask, 0.0
    safe = np.clip(np.where(np.isfinite(values), values, 0.0), -3.0, 5.0)
    legal = np.sort(safe[entry_mask])[::-1]
    width = max(1, min(int(consensus_actions), legal.size))
    trade_utility = float(np.mean(legal[:width]))
    relative = (
        trade_utility - float(safe[HOLD_ACTION]) - float(trade_margin_r)
    ) / max(.05, float(trade_temperature))
    trade_probability = 1.0 / (1.0 + float(np.exp(-np.clip(relative, -60.0, 60.0))))
    entry_logits = np.where(
        entry_mask,
        safe / max(.05, float(action_temperature)),
        -np.inf,
    )
    shifted = entry_logits - float(np.max(entry_logits[entry_mask]))
    conditional = np.where(entry_mask, np.exp(np.clip(shifted, -60.0, 60.0)), 0.0)
    conditional /= max(float(np.sum(conditional)), 1e-12)
    target[HOLD_ACTION] = 1.0 - trade_probability
    target[entry_mask] = trade_probability * conditional[entry_mask]
    return target, mask, trade_utility


def hierarchical_action_probabilities(logits, *, action_mask=None):
    """Return [P(HOLD), P(TRADE)*P(entry|TRADE)] on seven entry logits."""
    values = np.asarray(logits, dtype=float).reshape(-1)
    if values.size < 7:
        raise ValueError("hierarchical action probabilities require at least seven logits")
    mask = np.ones(7, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool).reshape(-1)[:7]
    if mask.size != 7:
        raise ValueError("hierarchical action mask must contain seven values")
    mask = mask.copy()
    mask[HOLD_ACTION] = True
    entry_mask = mask.copy()
    entry_mask[HOLD_ACTION] = False
    probabilities = np.zeros(values.size, dtype=float)
    if not np.any(entry_mask):
        probabilities[HOLD_ACTION] = 1.0
        return probabilities, 0.0, np.zeros(7, dtype=float)
    entry_values = values[:7]
    maximum = float(np.max(entry_values[entry_mask]))
    weights = np.where(entry_mask, np.exp(np.clip(entry_values - maximum, -60.0, 60.0)), 0.0)
    weight_sum = max(float(np.sum(weights)), 1e-12)
    conditional = weights / weight_sum
    log_mean_exp = maximum + float(np.log(weight_sum)) - float(np.log(np.count_nonzero(entry_mask)))
    trade_log_odds = float(np.clip(log_mean_exp - entry_values[HOLD_ACTION], -60.0, 60.0))
    trade_probability = 1.0 / (1.0 + float(np.exp(-trade_log_odds)))
    probabilities[HOLD_ACTION] = 1.0 - trade_probability
    probabilities[:7][entry_mask] = trade_probability * conditional[entry_mask]
    return probabilities, float(trade_probability), conditional


def hierarchical_policy_ascent(logits, selected_action: int, *, action_mask=None):
    """Exact ascent derivative of log pi(a) for the hierarchical policy."""
    probabilities, trade_probability, conditional = hierarchical_action_probabilities(
        logits, action_mask=action_mask
    )
    direction = np.zeros(np.asarray(logits).size, dtype=float)
    selected = int(selected_action)
    if selected == HOLD_ACTION:
        direction[HOLD_ACTION] = trade_probability
        direction[:7] -= trade_probability * conditional
    elif 0 < selected < 7 and conditional[selected] > 0.0:
        direction[HOLD_ACTION] = -(1.0 - trade_probability)
        direction[:7] -= trade_probability * conditional
        direction[selected] += 1.0
    return direction, probabilities, trade_probability


def hierarchical_preference_loss_and_ascent(logits, target, *, action_mask=None):
    """BCE(TRADE/HOLD) plus conditional-entry CE and its exact ascent."""
    values = np.asarray(logits, dtype=float).reshape(-1)
    desired = np.asarray(target, dtype=float).reshape(-1)
    if values.size < 7 or desired.size != 7:
        raise ValueError("hierarchical preference loss requires seven target values")
    probabilities, trade_probability, conditional = hierarchical_action_probabilities(
        values, action_mask=action_mask
    )
    trade_target = float(np.clip(np.sum(np.maximum(desired[1:7], 0.0)), 0.0, 1.0))
    conditional_target = np.zeros(7, dtype=float)
    if trade_target > 1e-12:
        conditional_target[1:7] = np.maximum(desired[1:7], 0.0) / trade_target
    direction = np.zeros(values.size, dtype=float)
    direction[HOLD_ACTION] = trade_probability - trade_target
    direction[:7] += (trade_target - trade_probability) * conditional
    direction[:7] += trade_target * (conditional_target - conditional)
    group_loss = -(
        trade_target * np.log(max(trade_probability, 1e-12))
        + (1.0 - trade_target) * np.log(max(1.0 - trade_probability, 1e-12))
    )
    conditional_loss = -trade_target * float(np.sum(
        conditional_target * np.log(np.maximum(conditional, 1e-12))
    ))
    return float(group_loss + conditional_loss), direction, probabilities, trade_probability


def timing_gate_loss_and_ascent(logits, enter_now_target:float, *, action_mask=None):
    """BCE for ENTER-NOW versus WAIT without changing entry-style ranking."""
    values=np.asarray(logits,dtype=float).reshape(-1)
    probabilities,trade_probability,conditional=hierarchical_action_probabilities(
        values,action_mask=action_mask
    )
    target=float(np.clip(enter_now_target,0.,1.))
    direction=np.zeros(values.size,dtype=float)
    # Exact ascent direction for the group BCE.  Every entry receives the same
    # aggregate pressure through its current conditional probability, so this
    # loss cannot turn BUY into SELL or MARKET into STOP/LIMIT.
    direction[HOLD_ACTION]=trade_probability-target
    direction[:len(ENTRY_PREFERENCE_ACTIONS)]+=(target-trade_probability)*conditional
    loss=-(
        target*np.log(max(trade_probability,1e-12))
        +(1.-target)*np.log(max(1.-trade_probability,1e-12))
    )
    return float(loss),direction,probabilities,float(trade_probability)


def conditional_entry_loss_and_ascent(logits,target,*,action_mask=None):
    """Train P(action | TRADE) without moving the HOLD/TRADE boundary."""
    values=np.asarray(logits,dtype=float).reshape(-1)
    labels=np.asarray(target,dtype=float).reshape(-1)
    if values.size<7 or labels.size!=7:
        raise ValueError("conditional entry loss requires seven entry logits/targets")
    mask=np.ones(7,dtype=bool) if action_mask is None else np.asarray(action_mask,dtype=bool).reshape(-1)[:7]
    if mask.size!=7:raise ValueError("conditional entry mask must contain seven values")
    entry_mask=mask.copy();entry_mask[HOLD_ACTION]=False
    direction=np.zeros(values.size,dtype=float);probabilities=np.zeros(7,dtype=float)
    if not np.any(entry_mask):return 0.,direction,probabilities
    safe_target=np.where(entry_mask,np.maximum(labels,0.),0.)
    if float(np.sum(safe_target))<=1e-12:return 0.,direction,probabilities
    safe_target/=float(np.sum(safe_target))
    shifted=np.where(entry_mask,values[:7],-np.inf)
    shifted-=float(np.max(shifted[entry_mask]))
    probabilities=np.where(entry_mask,np.exp(np.clip(shifted,-60.,60.)),0.)
    probabilities/=max(float(np.sum(probabilities)),1e-12)
    direction[:7]=safe_target-probabilities;direction[HOLD_ACTION]=0.
    loss=-float(np.sum(safe_target*np.log(np.maximum(probabilities,1e-12))))
    return loss,direction,probabilities
