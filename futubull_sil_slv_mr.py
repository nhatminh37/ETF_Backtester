class Strategy(StrategyBase):
    """
    Simple mean-reversion on SIL / SLV price ratio.

    Signal (1m bars):
      ratio = SIL / SLV
      z     = (ratio - mean(ratio, vol_window)) / std(ratio, vol_window)

      z > +entry_z  ->  Short SIL / Long SLV   (SIL rich vs silver)
      z < -entry_z  ->  Long SIL / Short SLV   (SIL cheap vs silver)

    Exit when |z| < exit_z, or take-profit, or time stop, or EOD.

    FUTUBULL BACKTEST:
      Trigger 1 = US.SIL  (1m candle at close)
      Trigger 2 = US.SLV  (1m candle at close)
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
        self.trig_symbol_sil = declare_trig_symbol()  # SIL
        self.trig_symbol_slv = declare_trig_symbol()  # SLV

    def global_variables(self):
        self.entry_z = show_variable(2.0, GlobalType.FLOAT)
        self.exit_z = show_variable(0.5, GlobalType.FLOAT)
        self.vol_window = show_variable(60, GlobalType.INT)
        self.hold_minutes = show_variable(30, GlobalType.INT)
        self.tp_pct = show_variable(0.8, GlobalType.FLOAT)
        self.notional = show_variable(10000, GlobalType.INT)
        self.skip_bars = show_variable(30, GlobalType.INT)
        self.eod_bars = show_variable(385, GlobalType.INT)
        self.alert_on_signal = show_variable(1, GlobalType.INT)
        self.bar = BarType.M1

    def custom_indicator(self):
        pass

    def _init_state(self):
        self.z = 0.0
        self.ratio = 0.0
        self.ratio_hist = []
        self.session_bars = 0
        self.in_position = 0
        self.direction = 0
        self.entry_sil = 0.0
        self.entry_slv = 0.0
        self.shares_sil = 0
        self.shares_slv = 0
        self.bars_held = 0
        self.last_bar_key = ""

    def handle_data(self):
        sil = current_price(symbol=self.trig_symbol_sil, price_type=THType.ALL)
        slv = current_price(symbol=self.trig_symbol_slv, price_type=THType.ALL)
        if sil is None or slv is None:
            return
        if sil <= 0 or slv <= 0:
            return

        bar_key = str(sil) + "|" + str(slv)
        if bar_key == self.last_bar_key:
            return
        self.last_bar_key = bar_key

        if self.session_bars >= int(self.eod_bars):
            self.session_bars = 0
            self.ratio_hist = []

        self.ratio = sil / slv
        self._append_ratio(self.ratio)
        self.z = self._zscore()
        self.session_bars = self.session_bars + 1

        if self.in_position == 1:
            self._manage_position(sil, slv)
            return

        if self.session_bars <= int(self.skip_bars):
            return
        if self.session_bars >= int(self.eod_bars):
            return
        if len(self.ratio_hist) < int(self.vol_window):
            return

        self._maybe_enter(sil, slv)

    def _append_ratio(self, r):
        self.ratio_hist.append(r)
        max_n = int(self.vol_window)
        if max_n < 20:
            max_n = 20
        while len(self.ratio_hist) > max_n:
            self.ratio_hist.pop(0)

    def _mean(self, vals):
        n = len(vals)
        if n < 1:
            return 0.0
        s = 0.0
        i = 0
        while i < n:
            s = s + vals[i]
            i = i + 1
        return s / n

    def _std(self, vals):
        n = len(vals)
        if n < 2:
            return 0.0
        m = self._mean(vals)
        var = 0.0
        i = 0
        while i < n:
            d = vals[i] - m
            var = var + d * d
            i = i + 1
        var = var / (n - 1)
        if var <= 0.0:
            return 0.0
        return var ** 0.5

    def _zscore(self):
        need = int(self.vol_window)
        if len(self.ratio_hist) < need:
            return 0.0
        m = self._mean(self.ratio_hist)
        sd = self._std(self.ratio_hist)
        if sd <= 0.0:
            return 0.0
        return (self.ratio - m) / sd

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

    def _maybe_enter(self, sil, slv):
        if abs(self.z) <= float(self.entry_z):
            return

        # z > 0: SIL rich -> short SIL / long SLV
        if self.z > 0:
            direction = 1
        else:
            direction = -1

        qty_sil = self._target_qty(sil)
        qty_slv = self._target_qty(slv)

        if direction > 0:
            qty_sil = self._cap_short_qty(self.trig_symbol_sil, qty_sil)
            qty_slv = self._cap_buy_qty(self.trig_symbol_slv, qty_slv)
        else:
            qty_sil = self._cap_buy_qty(self.trig_symbol_sil, qty_sil)
            qty_slv = self._cap_short_qty(self.trig_symbol_slv, qty_slv)

        if qty_sil < 1 or qty_slv < 1:
            if int(self.alert_on_signal) == 1:
                alert(content="SKIP qty SIL={} SLV={} z={:.2f}".format(qty_sil, qty_slv, self.z))
            return

        self._open_pair(direction, qty_sil, qty_slv)
        self.in_position = 1
        self.direction = direction
        self.entry_sil = sil
        self.entry_slv = slv
        self.shares_sil = qty_sil
        self.shares_slv = qty_slv
        self.bars_held = 0

        if int(self.alert_on_signal) == 1:
            alert(content="ENTRY z={:.2f} ratio={:.4f} dir={} SIL={} SLV={}".format(
                self.z, self.ratio, direction, qty_sil, qty_slv))

    def _manage_position(self, sil, slv):
        self.bars_held = self.bars_held + 1
        d = self.direction
        pnl_sil = d * float(self.shares_sil) * (self.entry_sil - sil)
        pnl_slv = d * float(self.shares_slv) * (slv - self.entry_slv)
        pnl_pct = 100.0 * (pnl_sil + pnl_slv) / float(self.notional)

        need_exit = 0
        reason = ""
        if abs(self.z) <= float(self.exit_z):
            need_exit = 1
            reason = "ZEXIT"
        elif pnl_pct >= float(self.tp_pct):
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
            alert(content="EXIT {} z={:.2f} pnl%={:.2f}".format(reason, self.z, pnl_pct))

        self.in_position = 0
        self.direction = 0
        self.bars_held = 0
        self.shares_sil = 0
        self.shares_slv = 0

    def _open_pair(self, direction, qty_sil, qty_slv):
        if direction > 0:
            place_market(symbol=self.trig_symbol_sil, qty=qty_sil, side=OrderSide.SELL)
            place_market(symbol=self.trig_symbol_slv, qty=qty_slv, side=OrderSide.BUY)
        else:
            place_market(symbol=self.trig_symbol_sil, qty=qty_sil, side=OrderSide.BUY)
            place_market(symbol=self.trig_symbol_slv, qty=qty_slv, side=OrderSide.SELL)

    def _close_pair(self):
        if self.direction > 0:
            qty_a = self._cap_buy_qty(self.trig_symbol_sil, self.shares_sil)
            qty_b = self._cap_sell_qty(self.trig_symbol_slv, self.shares_slv)
            if qty_a >= 1:
                place_market(symbol=self.trig_symbol_sil, qty=qty_a, side=OrderSide.BUY)
            if qty_b >= 1:
                place_market(symbol=self.trig_symbol_slv, qty=qty_b, side=OrderSide.SELL)
        else:
            qty_a = self._cap_sell_qty(self.trig_symbol_sil, self.shares_sil)
            qty_b = self._cap_buy_qty(self.trig_symbol_slv, self.shares_slv)
            if qty_a >= 1:
                place_market(symbol=self.trig_symbol_sil, qty=qty_a, side=OrderSide.SELL)
            if qty_b >= 1:
                place_market(symbol=self.trig_symbol_slv, qty=qty_b, side=OrderSide.BUY)
