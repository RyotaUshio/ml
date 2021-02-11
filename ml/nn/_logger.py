import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import dataclasses
import warnings
from typing import Sequence, Callable

from ..exceptions import NoImprovement


@dataclasses.dataclass
class logger:
    """学習経過の記録と損失のグラフの描画, および早期終了の制御
    """
    
    net : 'mlp'                  = dataclasses.field(default=None, repr=False)
    loss: Sequence[float]      = dataclasses.field(default_factory=list, repr=False)
    n_sample : int             = None
    batch_size: int            = None
    X_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    T_train : np.ndarray       = dataclasses.field(default=None, repr=False)
    X_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    T_val : np.ndarray         = dataclasses.field(default=None, repr=False)
    _validate : bool   = dataclasses.field(default=False, repr=False)
    val_loss: Sequence[float]  = dataclasses.field(default_factory=list, repr=False)
    color : str                = dataclasses.field(default='tab:blue', repr=False)
    color2 : str               = dataclasses.field(default='tab:orange', repr=False)
    how: str                   = dataclasses.field(default='plot', repr=False)
    delta_epoch : int          = dataclasses.field(default=10, repr=False)
    early_stopping: bool       = dataclasses.field(default=True)
    patience_epoch: int        = 10
    tol: float                 = 1e-5
    best_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    stop_params: dict          = dataclasses.field(init=False, default=None, repr=False)
    AIC : float                = dataclasses.field(init=False, default=None)
    BIC : float                = dataclasses.field(init=False, default=None)
    time: float                = None
    optimizer: '_optimizer_base' = None
    callback : Callable        = dataclasses.field(default=None, repr=False)

    def __post_init__(self):
        self.accumulated_loss = 0
        
        # whether to compute the values of loss for validation set (X_val, T_val)
        if not((self.X_val is None) and (self.T_val is None)):
            self._validate = True
            self.val_accuracy = []
                
        self.iterations = 0  # iterations so far
        self.epoch      = -1 # epochs so far
        # numbers of iterations per epoch
        self._iter_per_epoch = int(np.ceil(self.n_sample / self.batch_size))
        
        # how to show the values of loss function
        if self.how == 'both':
            self._plot, self._stdout = True, True
        elif self.how == 'plot':
            self._plot, self._stdout = True, False
        elif self.how == 'stdout':
            self._plot, self._stdout = False, True
        elif self.how == 'off':
            self._plot, self._stdout = False, False
        else:
            raise ValueError("logger.how must be either of the followings: 'both', 'plot', 'stdout' or 'off'")

        # Graph of loss
        if self._plot:
            self.fig, self.ax, self.secax = self.init_plot()
            plt.ion()

        # Early Stopping
        self.best_loss = np.inf
        if self._validate:
            self.best_val_loss = np.inf

        # start time of training
        self._t0 = time.time()
    
    def init_plot(self):
        fig, ax = plt.subplots(constrained_layout=True)        
        ax.set(xlabel='epochs', ylabel='loss')
        if self.iterations == 0:
            ax.set_xlim(0, self.delta_epoch)
            ax.set_ylim(0, 1)
        ax.grid(axis='y', linestyle='--')
        if self._validate and hasattr(self.net, 'test'):
            secax = ax.secondary_yaxis('right', functions=(self._to_percent, self._from_percent))
            secax.set_ylabel('accuracy [%]')
        else:
            secax = None
        
        return fig, ax, secax

    def _plot_every_epoch(self, logstr):            
        if self.epoch % self.delta_epoch == 1:
            self.ax.set_xlim(0, self.epoch + self.delta_epoch - 1)
            if self._validate:
                max_loss = max(self.loss + self.val_loss)
                if hasattr(self.net, 'test'):
                    max_loss = 0
            else:
                max_loss = max(self.loss)
            self.ax.set_ylim(0, max(max_loss, 1))

        x = [self.epoch - 1, self.epoch]
            
        if self._validate:
            self.ax.plot(x, self.val_loss[-2:], c=self.color2, label='validation loss')
            if hasattr(self.net, 'test'):
                self.ax.plot(self.val_accuracy, c=self.color2, linestyle='--', label='validation accuracy')
                
        self.ax.plot(x, self.loss[-2:], c=self.color, label='training loss')
        self.ax.set_title(logstr, fontsize=8)

        plt.show()
        plt.pause(0.2)

    def _plot_legend(self):
        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0]
        if self._validate:
            order = [1, 0]
            if hasattr(self, 'val_accuracy'):
                if self.val_accuracy:
                    order = [2, 0, 1]
                    
        plt.legend(
            [handles[idx] for idx in order], [labels[idx] for idx in order],
            bbox_to_anchor=(1, 0.5), loc='upper right', borderaxespad=1
        )
    
    def __call__(self):
        T_mini = self.net[-1].z - self.net[-1].delta
        self.accumulated_loss += self.net.loss(None, T_mini)
        
        # at the top of epoch
        if ((self.iterations >= self._iter_per_epoch)
            and (self.iterations % self._iter_per_epoch == 0)):
            
            self.epoch += 1
            self.loss.append(self.accumulated_loss / self._iter_per_epoch)
            self.accumulated_loss = 0

            # loss for the validation set (X_val, T_val)
            if self._validate:
                if self.net.dropout:
                    # if dropout is on, turn it off temporarily
                    self.net._set_training_flag(False)
                    
                self.val_loss.append(self.net.loss(self.X_val, self.T_val))
                if hasattr(self.net, 'test'):
                    accuracy = self.net.test(self.X_val, self.T_val, False)
                    self.val_accuracy.append(accuracy)
                
                if self.net.dropout:
                    self.net._set_training_flag(True)

            # log output
            logstr = f"Epoch {self.epoch}: Loss={self.loss[-1]:.3e}"
            if self._validate:
                logstr += f" (training), {self.val_loss[-1]:.3e} (validation)"
                if self.val_accuracy:
                    logstr += f", Accuracy={self.val_accuracy[-1]*100:.2f}% (validation)"
            
            if self._stdout:
                print(logstr)

            if self._plot:
                if self.epoch >= 1:    
                    self._plot_every_epoch(logstr)
                if self.epoch == 1:
                    self._plot_legend()

            # Early Stopping
            last_loss = self.loss[-1]            
            if self._validate:
                last_val_loss = self.val_loss[-1]
                # 検証用データに対する損失に一定以上の改善が見られない場合
                if last_val_loss > (self.best_val_loss - self.tol):
                    self._no_improvement_epoch += 1
                else:
                    self._no_improvement_epoch = 0
                # 現時点までの暫定最適値を更新(検証用データ) 
                if last_val_loss < self.best_val_loss:
                    self.best_val_loss = last_val_loss
                    self.best_params_val = self.net.get_params()
                    
            else:
                # 訓練データに対する損失に一定以上の改善が見られない場合
                if last_loss > (self.best_loss - self.tol):
                    self._no_improvement_epoch += 1
                else:
                    self._no_improvement_epoch = 0
            
            # 現時点までの暫定最適値を更新(訓練データ)
            if last_loss < self.best_loss:
                self.best_loss = last_loss
                self.best_params = self.net.get_params()

            if self._no_improvement_epoch > self.patience_epoch:
                which = 'Validation' if self._validate else 'Training'
                no_improvement_msg = (
                    f"{which} loss did not improve more than "
                    f"tol={self.tol} for the last {self.patience_epoch} epochs"
                    f" ({self.epoch} epochs so far)."
                )
                self.stop_params = self.net.get_params()
                if self.early_stopping:
                    raise NoImprovement(no_improvement_msg)

                warnings.warn(no_improvement_msg)

            # callback
            if self.callback is not None:
                self.callback(self.net)

        self.iterations += 1
        
    def end(self) -> None:
        # record time elapsed
        self._tf = time.time()
        self.time = self._tf - self._t0
        # calculate AIC & BIC (This is correct only when using (sigmoid, cross_entropy) or (softmax, multi_cross_entropy).)
        net = self.net
        from ._loss import cross_entropy
        if (isinstance(net.loss, cross_entropy)
            and
            net[-1].h.is_canonical):
            nll = net.loss(self.X_train, self.T_train) * self.n_sample # negative log likelihood
            n_param = 0                                                # number of parameters in net
            for layer in net[1:]:
                n_param += layer.W.size + layer.b.size
            self.AIC = 2 * nll + 2 * n_param
            self.BIC = 2 * nll + n_param * np.log(self.n_sample)
    
    def plot(self, color='tab:blue', color2='tab:orange', *args, **kwargs):
        """学習のあと、グラフをふたたび表示
        """
        fig, ax, secax = self.init_plot()
        
        if self.val_loss:
            ax.plot(self.val_loss, color=color2, label='valid.loss', *args, **kwargs)
            ax.plot(self.val_accuracy, c=self.color2, linestyle='--', label='valid.accuracy')

        ax.plot(self.loss, color=color, label='train.loss', *args, **kwargs)
        ax.set_ylim(bottom=0)
        
        self._plot_legend()
        return fig, ax, secax

    # __________________________ for pickle _________________________
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('fig', None)
        state.pop('ax', None)
        state.pop('secax', None)
        state.pop('callback', None)
        return state

    def save(self, filename):
        utils.save(self, filename)


    # ________________________ for init_plot() _______________________
    @classmethod
    def _to_percent(cls, x):
        return 100 * x
    
    @classmethod
    def _from_percent(cls, x):
        return 0.01 * x

