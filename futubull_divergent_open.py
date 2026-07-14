class Strategy(StrategyBase):
    """
    Divergent fade + per-leg TP 0.1%.

    Signal (prior day close-to-close):
      r_PAAS > 0 and r_SLV < 0  →  SHORT PAAS / LONG SLV at open
      both_sides=1 also trades the opposite.

    Exit each leg independently when it moves tp_pct% in your favor:
      short leg: price <= entry * (1 - tp_pct/100)
      long leg:  price >= entry * (1 + tp_pct/100)
    Any remaining leg is flattened at EOD.

    BACKTEST:
      Trigger1=US.PAAS, Trigger2=US.SLV, 1m candle at close
      Margin + short ON, cash >= 40000
    """

    def initialize(self):
        declare_strategy_type(AlgoStrategyType.SECURITY)
        self.trigger_symbols()
        self.custom_indicator()
        self.global_variables()
        self._init_state()

    def trigger_symbols(self):
        self.trig_symbol_paas = declare_trig_symbol()
        self.trig_symbol_slv = declare_trig_symbol()

    def global_variables(self):
        self.notional = show_variable(10000, GlobalType.INT)
        self.tp_pct = show_variable(0.1, GlobalType.FLOAT)
        self.both_sides = show_variable(0, GlobalType.INT)
        self.skip_bars = show_variable(1, GlobalType.INT)
        self.eod_bars = show_variable(385, GlobalType.INT)
        self.alert_on_signal = show_variable(1, GlobalType.INT)
        self.bar = BarType.M1

    def custom_indicator(self):
        pass

    def _init_state(self):
        self.prev_close_paas = 0.0
        self.prev_close_slv = 0.0
        self.has_prev_close = 0
        self.signal_ret_paas = 0.0
        self.signal_ret_slv = 0.0
        self.has_signal = 0
        self.session_bars = 0
        self.entered_today = 0
        self.in_position = 0
        self.direction = 0
        self.shares_paas = 0
        self.shares_slv = 0
        self.entry_paas = 0.0
        self.entry_slv = 0.0
        self.open_paas_leg = 0
        self.open_slv_leg = 0
        self.last_bar_key = ""

    def handle_data(self):
        paas = current_price(symbol=self.trig_symbol_paas, price_type=THType.ALL)
        slv = current_price(symbol=self.trig_symbol_slv, price_type=THType.ALL)
        if paas is None or slv is None:
            return
        if paas <= 0 or slv <= 0:
            return

        bar_key = str(paas) + "|" + str(slv)
        if bar_key == self.last_bar_key:
            return
        self.last_bar_key = bar_key

        if self.session_bars >= int(self.eod_bars):
            self._end_of_day(paas, slv)
            self.session_bars = 0
            self.entered_today = 0

        self.session_bars = self.session_bars + 1

        if self.in_position == 1:
            self._manage_legs(paas, slv)
            if self.session_bars >= int(self.eod_bars) - 1:
                self._flatten_remaining(paas, slv)
            return

        if self.entered_today == 0 and self.session_bars > int(self.skip_bars):
            self._maybe_enter(paas, slv)

    def _end_of_day(self, paas, slv):
        if self.has_prev_close == 1:
            self.signal_ret_paas = paas / self.prev_close_paas - 1.0
            self.signal_ret_slv = slv / self.prev_close_slv - 1.0
            self.has_signal = 1
        self.prev_close_paas = paas
        self.prev_close_slv = slv
        self.has_prev_close = 1
        if self.in_position == 1:
            self._flatten_remaining(paas, slv)

    def _signal_direction(self):
        if self.has_signal != 1:
            return 0
        ra = float(self.signal_ret_paas)
        rb = float(self.signal_ret_slv)
        if ra > 0.0 and rb < 0.0:
            return 1
        if int(self.both_sides) == 1 and ra < 0.0 and rb > 0.0:
            return -1
        return 0

    def _maybe_enter(self, paas, slv):
        direction = self._signal_direction()
        if direction == 0:
            self.entered_today = 1
            return

        qty_paas = self._target_qty(paas)
        qty_slv = self._target_qty(slv)

        if direction > 0:
            qty_paas = self._cap_short_qty(self.trig_symbol_paas, qty_paas)
            qty_slv = self._cap_buy_qty(self.trig_symbol_slv, qty_slv)
        else:
            qty_paas = self._cap_buy_qty(self.trig_symbol_paas, qty_paas)
            qty_slv = self._cap_short_qty(self.trig_symbol_slv, qty_slv)

        self.entered_today = 1
        if qty_paas < 1 or qty_slv < 1:
            if int(self.alert_on_signal) == 1:
                alert(content="SKIP qty PAAS={} SLV={}".format(qty_paas, qty_slv))
            return

        self._open_pair(direction, qty_paas, qty_slv)
        self.in_position = 1
        self.direction = direction
        self.shares_paas = qty_paas
        self.shares_slv = qty_slv
        self.entry_paas = paas
        self.entry_slv = slv
        self.open_paas_leg = 1
        self.open_slv_leg = 1

        if int(self.alert_on_signal) == 1:
            alert(content="ENTRY dir={} rP={:.3f} rS={:.3f}".format(
                direction, self.signal_ret_paas, self.signal_ret_slv))

    def _manage_legs(self, paas, slv):
        thr = float(self.tp_pct) / 100.0
        d = self.direction

        # Close PAAS leg on TP
        if self.open_paas_leg == 1:
            hit = 0
            if d > 0:
                # short PAAS
                if paas <= self.entry_paas * (1.0 - thr):
                    hit = 1
            else:
                # long PAAS
                if paas >= self.entry_paas * (1.0 + thr):
                    hit = 1
            if hit == 1:
                self._close_paas_leg()
                self.open_paas_leg = 0
                self.shares_paas = 0
                if int(self.alert_on_signal) == 1:
                    alert(content="TP PAAS @ {}".format(paas))

        # Close SLV leg on TP
        if self.open_slv_leg == 1:
            hit = 0
            if d > 0:
                # long SLV
                if slv >= self.entry_slv * (1.0 + thr):
                    hit = 1
            else:
                # short SLV
                if slv <= self.entry_slv * (1.0 - thr):
                    hit = 1
            if hit == 1:
                self._close_slv_leg()
                self.open_slv_leg = 0
                self.shares_slv = 0
                if int(self.alert_on_signal) == 1:
                    alert(content="TP SLV @ {}".format(slv))

        if self.open_paas_leg == 0 and self.open_slv_leg == 0:
            self.in_position = 0
            self.direction = 0

    def _flatten_remaining(self, paas, slv):
        if self.open_paas_leg == 1:
            self._close_paas_leg()
            self.open_paas_leg = 0
            self.shares_paas = 0
        if self.open_slv_leg == 1:
            self._close_slv_leg()
            self.open_slv_leg = 0
            self.shares_slv = 0
        if self.in_position == 1:
            if int(self.alert_on_signal) == 1:
                alert(content="EXIT EOD flatten")
        self.in_position = 0
        self.direction = 0

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

    def _open_pair(self, direction, qty_paas, qty_slv):
        if direction > 0:
            place_market(symbol=self.trig_symbol_paas, qty=qty_paas, side=OrderSide.SELL)
            place_market(symbol=self.trig_symbol_slv, qty=qty_slv, side=OrderSide.BUY)
        else:
            place_market(symbol=self.trig_symbol_paas, qty=qty_paas, side=OrderSide.BUY)
            place_market(symbol=self.trig_symbol_slv, qty=qty_slv, side=OrderSide.SELL)

    def _close_paas_leg(self):
        if self.shares_paas < 1:
            return
        if self.direction > 0:
            qty = self._cap_buy_qty(self.trig_symbol_paas, self.shares_paas)
            if qty >= 1:
                place_market(symbol=self.trig_symbol_paas, qty=qty, side=OrderSide.BUY)
        else:
            qty = self._cap_sell_qty(self.trig_symbol_paas, self.shares_paas)
            if qty >= 1:
                place_market(symbol=self.trig_symbol_paas, qty=qty, side=OrderSide.SELL)

    def _close_slv_leg(self):
        if self.shares_slv < 1:
            return
        if self.direction > 0:
            qty = self._cap_sell_qty(self.trig_symbol_slv, self.shares_slv)
            if qty >= 1:
                place_market(symbol=self.trig_symbol_slv, qty=qty, side=OrderSide.SELL)
        else:
            qty = self._cap_buy_qty(self.trig_symbol_slv, self.shares_slv)
            if qty >= 1:
                place_market(symbol=self.trig_symbol_slv, qty=qty, side=OrderSide.BUY)
