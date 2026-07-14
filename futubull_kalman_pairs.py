class Strategy(StrategyBase):
    """
    Simple PAAS vs AG relative-strength fade (no Kalman, no z-score).

    Every 1m bar after warm-up:
      rel = sum(r_PAAS - r_AG) over last lookback bars
      if rel >  +entry_pct%  -> Short PAAS / Long AG
      if rel <  -entry_pct%  -> Long PAAS / Short AG

    Defaults are loose so you should see fills quickly.

    BACKTEST:
      Trigger1 = US.PAAS, Trigger2 = US.AG, both 1m candle at close
      Margin + short selling ON
      Cash >= 40000
    """

    def initialize(self):
        declare_strategy_type(AlgoStrategyType.SECURITY)
        self.trigger_symbols()
        self.custom_indicator()
        self.global_variables()
        self._init_state()

    def trigger_symbols(self):
        self.trig_symbol_paas = declare_trig_symbol()
        self.trig_symbol_ag = declare_trig_symbol()

    def global_variables(self):
        # Entry when PAAS outperformed AG by this % over lookback minutes
        self.entry_pct = show_variable(0.3, GlobalType.FLOAT)
        self.lookback = show_variable(5, GlobalType.INT)
        self.hold_minutes = show_variable(10, GlobalType.INT)
        self.tp_pct = show_variable(0.5, GlobalType.FLOAT)
        self.notional = show_variable(10000, GlobalType.INT)
        self.skip_bars = show_variable(15, GlobalType.INT)
        self.eod_bars = show_variable(385, GlobalType.INT)
        self.alert_on_signal = show_variable(1, GlobalType.INT)
        self.bar = BarType.M1

    def custom_indicator(self):
        pass

    def _init_state(self):
        self.prev_paas = 0.0
        self.prev_ag = 0.0
        self.has_prev = 0
        self.session_bars = 0
        self.diffs = []
        self.in_position = 0
        self.direction = 0
        self.entry_paas = 0.0
        self.entry_ag = 0.0
        self.shares_paas = 0
        self.shares_ag = 0
        self.bars_held = 0
        self.last_bar_key = ""

    def handle_data(self):
        paas = current_price(symbol=self.trig_symbol_paas, price_type=THType.ALL)
        ag = current_price(symbol=self.trig_symbol_ag, price_type=THType.ALL)
        if paas is None or ag is None:
            return
        if paas <= 0 or ag <= 0:
            return

        bar_key = str(round(paas, 4)) + "|" + str(round(ag, 4))
        if bar_key == self.last_bar_key:
            return
        self.last_bar_key = bar_key

        # Soft day rollover by bar count (ignore overnight return)
        if self.session_bars >= int(self.eod_bars):
            self.session_bars = 0
            self.diffs = []
            self.prev_paas = paas
            self.prev_ag = ag
            self.has_prev = 1
            self.session_bars = 1
            return

        if self.has_prev == 1:
            r_paas = paas / self.prev_paas - 1.0
            r_ag = ag / self.prev_ag - 1.0
            # Skip huge overnight jumps without resetting wrongly mid-day
            if abs(r_paas) < 0.05 and abs(r_ag) < 0.05:
                self.diffs.append(r_paas - r_ag)
                max_n = int(self.lookback)
                if max_n < 2:
                    max_n = 2
                while len(self.diffs) > max_n:
                    self.diffs.pop(0)

        self.prev_paas = paas
        self.prev_ag = ag
        self.has_prev = 1
        self.session_bars = self.session_bars + 1

        if self.in_position == 1:
            self._manage_position(paas, ag)
            return

        if self.session_bars <= int(self.skip_bars):
            return
        if self.session_bars >= int(self.eod_bars):
            return
        if len(self.diffs) < int(self.lookback):
            return

        self._maybe_enter(paas, ag)

    def _rel_pct(self):
        s = 0.0
        i = 0
        while i < len(self.diffs):
            s = s + self.diffs[i]
            i = i + 1
        return 100.0 * s

    def _target_qty(self, price):
        qty = int(float(self.notional) / price)
        if qty < 1:
            qty = 1
        return qty

    def _cap_buy_qty(self, symbol, qty):
        avail = max_qty_to_buy_on_margin(symbol=symbol, order_type=OrdType.MKT)
        if avail is None or avail < 1:
            avail = max_qty_to_buy_on_cash(symbol=symbol, order_type=OrdType.MKT)
        if avail is None or avail < 1:
            return 0
        if qty > avail:
            return int(avail)
        return qty

    def _cap_short_qty(self, symbol, qty):
        avail = max_qty_to_sell_short(symbol=symbol, order_type=OrdType.MKT)
        if avail is None or avail < 1:
            return 0
        if qty > avail:
            return int(avail)
        return qty

    def _cap_sell_qty(self, symbol, qty):
        avail = max_qty_to_sell(symbol=symbol)
        if avail is None or avail < 1:
            return 0
        if qty > avail:
            return int(avail)
        return qty

    def _maybe_enter(self, paas, ag):
        rel = self._rel_pct()
        thr = float(self.entry_pct)
        direction = 0
        if rel > thr:
            direction = 1   # PAAS strong -> short PAAS / long AG
        elif rel < -thr:
            direction = -1  # PAAS weak  -> long PAAS / short AG
        else:
            return

        qty_paas = self._target_qty(paas)
        qty_ag = self._target_qty(ag)

        if direction > 0:
            qty_paas = self._cap_short_qty(self.trig_symbol_paas, qty_paas)
            qty_ag = self._cap_buy_qty(self.trig_symbol_ag, qty_ag)
        else:
            qty_paas = self._cap_buy_qty(self.trig_symbol_paas, qty_paas)
            qty_ag = self._cap_short_qty(self.trig_symbol_ag, qty_ag)

        if qty_paas < 1 or qty_ag < 1:
            if int(self.alert_on_signal) == 1:
                alert(content="SKIP qty PAAS={} AG={} rel={:.3f}".format(qty_paas, qty_ag, rel))
            return

        self._open_pair(direction, qty_paas, qty_ag)
        self.in_position = 1
        self.direction = direction
        self.entry_paas = paas
        self.entry_ag = ag
        self.shares_paas = qty_paas
        self.shares_ag = qty_ag
        self.bars_held = 0
        self.diffs = []

        if int(self.alert_on_signal) == 1:
            alert(content="ENTRY rel={:.3f}% dir={} PAAS={} AG={}".format(
                rel, direction, qty_paas, qty_ag))

    def _manage_position(self, paas, ag):
        self.bars_held = self.bars_held + 1
        d = self.direction
        pnl_paas = d * float(self.shares_paas) * (self.entry_paas - paas)
        pnl_ag = d * float(self.shares_ag) * (ag - self.entry_ag)
        pnl_pct = 100.0 * (pnl_paas + pnl_ag) / float(self.notional)

        need_exit = 0
        reason = ""
        if pnl_pct >= float(self.tp_pct):
            need_exit = 1
            reason = "TP"
        elif self.bars_held >= int(self.hold_minutes):
            need_exit = 1
            reason = "TIME"
        elif self.session_bars >= int(self.eod_bars) - 1:
            need_exit = 1
            reason = "EOD"

        if need_exit == 0:
            return

        self._close_pair()
        if int(self.alert_on_signal) == 1:
            alert(content="EXIT {} pnl%={:.2f}".format(reason, pnl_pct))

        self.in_position = 0
        self.direction = 0
        self.bars_held = 0
        self.shares_paas = 0
        self.shares_ag = 0

    def _open_pair(self, direction, qty_paas, qty_ag):
        if direction > 0:
            place_market(symbol=self.trig_symbol_paas, qty=qty_paas, side=OrderSide.SELL)
            place_market(symbol=self.trig_symbol_ag, qty=qty_ag, side=OrderSide.BUY)
        else:
            place_market(symbol=self.trig_symbol_paas, qty=qty_paas, side=OrderSide.BUY)
            place_market(symbol=self.trig_symbol_ag, qty=qty_ag, side=OrderSide.SELL)

    def _close_pair(self):
        if self.direction > 0:
            qty_p = self._cap_buy_qty(self.trig_symbol_paas, self.shares_paas)
            qty_a = self._cap_sell_qty(self.trig_symbol_ag, self.shares_ag)
            if qty_p >= 1:
                place_market(symbol=self.trig_symbol_paas, qty=qty_p, side=OrderSide.BUY)
            if qty_a >= 1:
                place_market(symbol=self.trig_symbol_ag, qty=qty_a, side=OrderSide.SELL)
        else:
            qty_p = self._cap_sell_qty(self.trig_symbol_paas, self.shares_paas)
            qty_a = self._cap_buy_qty(self.trig_symbol_ag, self.shares_ag)
            if qty_p >= 1:
                place_market(symbol=self.trig_symbol_paas, qty=qty_p, side=OrderSide.SELL)
            if qty_a >= 1:
                place_market(symbol=self.trig_symbol_ag, qty=qty_a, side=OrderSide.BUY)
