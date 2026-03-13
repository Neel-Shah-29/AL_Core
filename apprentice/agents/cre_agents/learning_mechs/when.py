import numpy as np
from pathlib import Path
import warnings
import os
from abc import ABCMeta
from abc import abstractmethod
from cre.utils import PrintElapse
from .registers import register_when
from ....shared import ElapseLogger
from apprentice.agents.cre_agents.hint_nlp import build_hf_llm_call


def _when_weight_debug_enabled():
    return os.environ.get("CRE_STAND_WEIGHT_DEBUG", "0") == "1"




# ------------------------------------------------------------------------
# : BaseWhen

# TODO: COMMENTS
class BaseWhen(metaclass=ABCMeta):
    def __init__(self, skill,**kwargs):
        self.skill = skill
        self.agent = skill.agent
        self.check_sanity = kwargs.get('check_sanity', True)
        self._sanity_warned = False

        # Note this line makes it possible to call 
        super(BaseWhen, self).__init__(skill, **kwargs)

    def sanity_check_ifit(self, state, skill_app, reward):
        # SANITY CHECK: Classifier Inconsistencies 
        if(self.check_sanity and reward is not None):
            prediction = self.predict(state, skill_app.match)
            if(reward != prediction):
                if not self._sanity_warned:
                    warnings.warn(
                        "(Sanity Check Warning): When-learning mechanism for "
                        f"skill {self.skill} predicted {prediction:.2f} but "
                        f"received reward {reward:.2f}. This usually means the "
                        "current state/features cannot distinguish two similar "
                        "situations or there is a data-prep bug. "
                        "Continuing training; this warning is shown once per skill.",
                        RuntimeWarning,
                    )
                    self._sanity_warned = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        
        # Inject sanity checks after ifit
        ifit = cls.ifit
        def ifit_w_sanity_check(self, state, skill_app, reward):
            ifit(self, state, skill_app, reward)
            self.sanity_check_ifit(state, skill_app, reward)
        setattr(cls, 'ifit', ifit_w_sanity_check)

        # ifit_lgr = cls.ifit_elapse_logger = ElapseLogger("ifit")
        # _ifit = cls.ifit
        # def ifit_w_elog(self, state, skill_app, reward):
        #     with ifit_lgr:
        #         val = _ifit(self, state, skill_app, reward)
        #     return val
        # setattr(cls, 'ifit', ifit_w_elog)

        # predict_lgr = cls.predict_elapse_logger = ElapseLogger("predict")
        # predict = cls.predict
        # def predict_w_elog(self, state, match):
        #     with predict_lgr:
        #         val = predict(self, state, match)
        #     return val
        # setattr(cls, 'predict', predict_w_elog)

    def ifit(self, state, skill_app, reward):
        """
        
        :param state: 
        """
        raise NotImplemented()

    def fit(self, states, skill_apps, reward):
        """
        
        """
        raise NotImplemented()

    def score(self, state, skill_app):
        """
        
        """
        raise NotImplemented()

    def as_conditions(self):
        """
        
        """
        raise NotImplemented()

    def predict(self, state, match):
        """
        
        """
        raise NotImplemented()

    def get_info(self, **kwargs):
        return {}


# ------------------------------------------------------------------------
# : Helpful mixins for resusing code in when-learning mechanisms

# --------------------------------------------
# : RefittableMixin

class RefittableMixin():
    '''A mixin which implements add_example() and remove_example().
        self.examples maps skill_apps to tuples of (index, reward)
     '''
    def __init__(self, skill,**kwargs):
        self.examples = {}

    def transform(self, state, match):
        '''Should take in a state and skill_app and return '''
        raise NotImplemented()

    def insert_transformed(self, transformed_state, skill_app, reward, index):
        raise NotImplemented()

    def remove_index(self):
        raise NotImplemented()        

    def _assert_skill_app_from_state(self, state, skill_app):
        assert state.get('__uid__') == skill_app.state_uid, (
            f"Provided state {state.get('__uid__')} is not associated with skill app: {skill_app}"
        )

    def add_example(self, state, skill_app, reward):
        ''' Adds a new training example. Applies transform() and insert_transformed() 
            from the child class. Checks to see if the example is new or a repeat and 
            returns the new index or possibly old index for the example and the add 
            attempt changed the set of training examples.
        ''' 
        # self._assert_skill_app_from_state(state, skill_app)
        did_change = False
        index, old_reward = self.examples.get(skill_app,(-1,0))
        # if(index != -1): 
        #     print("REPLACING FEEDBACK", skill_app, old_reward, "->", reward, "@ index", index)

        new_index = index
        if(reward is None and index != -1):
            self.remove_example(state, skill_app)
            return -1
        elif(reward is not None):
            new_index = len(self.examples) if index == -1 else index

            # NOTE: temping to only re-insert on reward changes, but meta-features can make it helpful
            #  to retrain if possible new feautres. 
            self.examples[skill_app] = (new_index, reward)
            prev_active_skill_app = getattr(self, "_active_skill_app", None)
            self._active_skill_app = skill_app
            try:
                transformed_state = self.transform(state, skill_app.match)
            finally:
                self._active_skill_app = prev_active_skill_app
            


            self.insert_transformed(transformed_state, skill_app, reward, new_index)
                
        return new_index

    def remove_example(self, state, skill_app):
        ''' Attempt to remove a skill_app instance from the training set.
            Shifts the set of indicies and calls rebase_examples() from the
            child class.
        '''
        
        # raise ValueError("REMOVE EXAMPLE")
        # self._assert_skill_app_from_state(state, skill_app)
        did_change = False
        index, old_reward = self.examples.get(skill_app,(-1,0))
        if(index != -1):
            print("REMOVE EXAMPLE", skill_app)
            for skill_app, (ind, reward) in self.examples.items():
                if(ind > index):
                    self.examples[skill_app] = (ind-1, reward)
            del self.examples[skill_app]
            did_change = True
            self.remove_index(index)
            # self.rebase_examples()

        return index, did_change



# --------------------------------------------
# : VectorTransformMixin

class VectorTransformMixin(RefittableMixin):
    def __init__(self, skill, encode_relative=True, one_hot=False,
                encode_missing=True, rel_enc_min_sources=None,
                starting_state_format='flat_featurized',
                extra_features=[],
                 **kwargs):
        super().__init__(skill,**kwargs)
        self.starting_state_format = 'flat_featurized'
        self.encode_relative = encode_relative
        self.extra_features = extra_features
        self.one_hot = one_hot
        self.encode_missing = encode_missing
        self.rel_enc_min_sources = rel_enc_min_sources
        self.gated_hints = kwargs.get('gated_hints', False)
        self.gated_use_llm = kwargs.get('gated_use_llm', False)
        self.gated_filter_labels = kwargs.get('gated_filter_labels', True)
        self.gated_upweight = kwargs.get('gated_upweight', 5.0)
        self.gated_hint_map = kwargs.get('gated_hint_map', None)
        self.apply_ft_weights_to_nominal = kwargs.get('apply_ft_weights_to_nominal', False)
        self.hint_key_resolver = kwargs.get('hint_key_resolver', None)
        self.weight_debug = kwargs.get('weight_debug', _when_weight_debug_enabled())
        self.weighted_hint_only = kwargs.get('predict_with_weight_only', False)
        self._last_ft_weights = None
        self._active_skill_app = None

        agent = skill.agent

        # Initialize Vectorizer
        from numba.types import f8, i8, string, boolean
        from cre.transform import Vectorizer
        self.vectorizer = Vectorizer([f8, string, boolean], one_hot, self.encode_missing)

        if(self.agent.enumerizer):

            # Recovers original keys and values before enumerization and vectorization  
            def inv_mapper(key_ind, val_nom):
                from cre import Var, CREFunc
                from cre.tuple_fact import TupleFactProxy
                if(self.one_hot):
                    key, nom = self.vectorizer.unvectorize(key_ind)
                else:
                    key, nom = self.vectorizer.unvectorize(key_ind, val_nom)

                if(isinstance(key, TupleFactProxy)):
                    key_head = key[0]
                    if(isinstance(key_head, (Var, CREFunc))):
                        typ = key_head.return_type

                    # TODO: Perhaps there is a less hardcoded way of doing this.
                    elif(key_head == "SkillCand:" or key_head == "SkillValueCount:"):
                        typ = key[1].return_type
                else:
                    typ = key.return_type

                if((not self.one_hot or self.encode_missing) and nom == 0):
                    val = None
                else:
                    val = self.agent.enumerizer.from_enum(nom, typ)


                return False, key, val # TODO: 01/14/2025 (is this right?)
            self.inv_mapper = inv_mapper

        # Initialize or retrive relative_encoder
        if(encode_relative):
            from cre.transform import RelativeEncoder
            # TODO: Backup won't work without fact_types
            self.relative_encoder = RelativeEncoder(agent.fact_types)

        self.X_nom = np.empty((0,0), dtype=np.int64)
        self.Y = np.empty(0, dtype=np.int64)

    def _wdebug(self, msg):
        if self.weight_debug:
            skill_name = getattr(self.skill, "action_type", "unknown_skill")
            print(f"[WhenWeightDebug:{skill_name}] {msg}")

    def _resolve_gated_hint_key(self):
        if callable(self.hint_key_resolver):
            try:
                return self.hint_key_resolver(self.skill)
            except Exception:
                pass
        active_skill_app = getattr(self, "_active_skill_app", None)
        if active_skill_app is not None:
            try:
                active_action = getattr(active_skill_app, "action", None)
                active_selection = getattr(active_action, "selection", None)
                if isinstance(active_selection, str) and active_selection:
                    return active_selection
            except Exception:
                pass
        try:
            for skill_app in getattr(self.skill, "skill_apps", {}).values():
                action = getattr(skill_app, "action", None)
                selection = getattr(action, "selection", None)
                if isinstance(selection, str) and selection:
                    return selection
        except Exception:
            pass
        if getattr(self.skill, "label", None):
            return self.skill.label
        return getattr(self.skill, "action_type", None)

    def _compute_gated_weights(self, featurized_state):
        if not self.gated_hints:
            return None
        gated = None
        try:
            from tutor_gym.sandbox.geometry import run_al_copy_2 as gated
        except Exception:
            gated = None
        if gated is None:
            try:
                import importlib.util
                module_path = Path(__file__).resolve().parents[5] / "tutor_gym" / "sandbox" / "geometry" / "run_al_copy_2.py"
                spec = importlib.util.spec_from_file_location("run_al_copy_2", module_path)
                if spec and spec.loader:
                    gated = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(gated)
            except Exception:
                return None

        hint_key = self._resolve_gated_hint_key()
        hint_text = gated.resolve_hint_text(hint_key, hint_map=self.gated_hint_map)

        llm_call = None
        if self.gated_use_llm:
            try:
                llm_call = build_hf_llm_call()
            except Exception:
                llm_call = None

        predicates, gvals = gated.extract_predicates(featurized_state)
        if not predicates:
            self._wdebug("No predicates found; cannot compute gated feature weights.")
            return None

        indices = gated.select_hint_predicates(
            predicates,
            hint_text,
            llm_call=llm_call,
            filter_labels=self.gated_filter_labels,
        )
        selected_gvals = [gvals[i] for i in indices]
        _, weights = gated.build_weight_vector(
            self.vectorizer,
            selected_gvals,
            upweight=self.gated_upweight,
        )
        n_up = int(np.count_nonzero(weights > 1.0))
        self._wdebug(
            f"Computed weights for hint_key={hint_key!r}; selected={len(indices)} predicates, "
            f"upweighted_slots={n_up}, max_weight={float(np.max(weights)) if len(weights) else 0.0}"
        )
        return weights

    def transform(self, state, match):
        featurized_state = state.get("flat_featurized")

        # if(not self.encode_relative):
        featurized_state = featurized_state.copy()
        agent = self.skill.agent

        
        for extra_feature in self.extra_features:
            featurized_state = extra_feature(self, state, featurized_state, match)   ####


        wm = state.get("working_memory")
        if(self.encode_relative):
            self.relative_encoder.set_in_memset(wm)
            _vars = self.skill.where_lrn_mech._ensure_vars(match)
            # print(":::", self.skill.id_num, "_vars", _vars[0].base_ptr)

            # print(featurized_state)

            # Featurize state relative to selection
            #  NOTE: Could also use arguments, but there is currently a hard
            #  to find bug associated with this
            # print(match[0], type(match[0]))
            # print(featurized_state)
            sources = match
            if(self.rel_enc_min_sources is not None):
                # print("rel_enc_min_sources", self.rel_enc_min_sources)
                # Note: setting rel_enc_min_sources = 1 only uses 
                #  the first variable, which is typically the selection.
                sources = sources[:self.rel_enc_min_sources]
                _vars = _vars[:self.rel_enc_min_sources]

            featurized_state = self.relative_encoder.encode_relative_to(
                featurized_state, sources, _vars)
        # else:
        #     featurized_state = self.relative_encoder.encode_relative_to(
        #         featurized_state, [match[0]], [_vars[0]])
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvv")
        # print(featurized_state)
        # print("^^^^^^^^^^^^^^^^^^^^^^^^^^")
        # if(repr(self.skill.how_part) == "NumericalToStr(TensDigit(Add3(CastFloat(a.value), CastFloat(b.value), CastFloat(c.value))))" and
        #    "S_Qr9" in state.get('__uid__')):
            # print('--------------------------------------')
            # print(self.conds)
            # print(matches)
            # print('--------------------------------------')
            

        continuous, nominal = self.vectorizer(featurized_state)
        self._last_ft_weights = self._compute_gated_weights(featurized_state)
        if self._last_ft_weights is not None:
            n_up = int(np.count_nonzero(self._last_ft_weights > 1.0))
            self._wdebug(
                f"Transform produced nominal_len={len(nominal)} and weight_len={len(self._last_ft_weights)} "
                f"(upweighted_slots={n_up})"
            )
        if self.apply_ft_weights_to_nominal and self._last_ft_weights is not None:
            max_len = min(len(nominal), len(self._last_ft_weights))
            if max_len > 0:
                nominal = nominal.astype(np.int64, copy=True)
                nominal[:max_len] = nominal[:max_len] * self._last_ft_weights[:max_len].astype(np.int64)
        #### -------Print mapping------------####
        # print(self.skill)
        # print(self.vectorizer)
        # print(nominal)
        # print("---------------------------------------")
        # for (ind, val) in self.vectorizer.make_inv_map().items():
        #     print("*", ind, nominal[ind], ind,val)
        # ind_vals = sorted([(ind, str(val)) for (ind, val) in self.vectorizer.get_inv_map().items()],
        #                 key=lambda t : t[1])
        # for ind, val in ind_vals:
        #     print(ind, ":", nominal[ind], val)
        # print("---------------------------------------")
        #####
        return continuous, nominal

    def insert_transformed(self, transformed_state, skill_app, reward, index):
        continuous, nominal = transformed_state

        n, m = self.X_nom.shape
        new_shape = (max(n, index+1), max(m, len(nominal)))
        
        if(new_shape != self.X_nom.shape):
            # print("NEW SHAPE", new_shape, n, index+1, m, len(nominal))
            # Copy old data into new matrix
            new_X_nom = np.zeros(new_shape, dtype=np.int64)
            new_Y = np.zeros(new_shape[0], dtype=np.int64)
            new_X_nom[:n, :m] = self.X_nom
            new_Y[:n] = self.Y

            # Copy old data into new training matrix
            self.X_nom = new_X_nom
            self.Y = new_Y

        self.X_nom[index] = nominal
        self.Y[index] = reward
        # print(self.X_nom,  self.Y)

    def remove_index(self, index):
        self.X_nom = np.concatenate([self.X_nom[:index], self.X_nom[index+1:]])
        self.Y = np.concatenate([self.Y[:index], self.Y[index+1:]])

class BasicSKL(BaseWhen, VectorTransformMixin):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        # Note: Most Sklearn classifiers only work well with one-hot
        #  for instance decision trees use CART implementation
        kwargs['one_hot'] = kwargs.get('one_hot', True)
        BaseWhen.__init__(self, skill,**kwargs)
        VectorTransformMixin.__init__(self, skill, **kwargs)

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y

        with PrintElapse(f"{type(self).__name__} fit:"):
            self.classifier.fit(self.X_nom, self.Y) # Re-fit

    def remove(self, state, skill_app):
        self.remove_example(state, skill_app) # Insert into X_nom, Y
        self.classifier.fit(self.X_nom, None, self.Y) # Re-fit

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)
        prediction = self.classifier.predict(X_nom_subset)[0]        
        return prediction

class BasicSTAND(BaseWhen, VectorTransformMixin):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        # Note: Most STAND tends to work best without one-hot by default
        kwargs['one_hot'] = kwargs.get('one_hot', False)
        BaseWhen.__init__(self, skill,**kwargs)
        VectorTransformMixin.__init__(self, skill, **kwargs)

    def _get_ft_weights(self):
        ft_weights = getattr(self, "_last_ft_weights", None)
        if ft_weights is None:
            self._wdebug("No cached feature weights available for STAND fit.")
            return None
        if self.X_nom.size == 0:
            self._wdebug("X_nom is empty; skipping feature weights.")
            return None
        m = self.X_nom.shape[1]
        if m == 0:
            self._wdebug("X_nom has zero columns; skipping feature weights.")
            return None
        if len(ft_weights) < m:
            ft_weights = np.pad(ft_weights, (0, m - len(ft_weights)), constant_values=1.0)
        elif len(ft_weights) > m:
            ft_weights = ft_weights[:m]
        ft_weights = ft_weights.astype(np.float64)
        n_up = int(np.count_nonzero(ft_weights > 1.0))
        self._wdebug(
            f"Prepared STAND nominal weights len={len(ft_weights)}, upweighted_slots={n_up}, "
            f"max_weight={float(np.max(ft_weights)) if len(ft_weights) else 0.0}"
        )
        return ft_weights

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y
        if(len(self.X_nom) == 0): return

        nom_ft_weights = self._get_ft_weights()
        self._wdebug(
            f"Calling STAND.fit from ifit with X_nom_shape={self.X_nom.shape}, y_len={len(self.Y)}, "
            f"weights_present={nom_ft_weights is not None}"
        )
        self.classifier.fit(self.X_nom, None, self.Y, nom_ft_weights=nom_ft_weights) # Re-fit ####

    def fit(self, skill_app_reward_pairs):
        cover = set()
        old_apps = set(self.examples.keys())
        for skill_app, reward in skill_app_reward_pairs:
            state = skill_app.state
            cover.add(skill_app)
            self.add_example(state, skill_app, reward) # Insert into X_nom, Y

        not_cover = old_apps.difference(cover)
        for skill_app in not_cover:
            state = skill_app.state
            self.remove_example(state, skill_app)

        nom_ft_weights = self._get_ft_weights()
        self._wdebug(
            f"Calling STAND.fit from fit with X_nom_shape={self.X_nom.shape}, y_len={len(self.Y)}, "
            f"weights_present={nom_ft_weights is not None}"
        )
        self.classifier.fit(self.X_nom, None, self.Y, nom_ft_weights=nom_ft_weights) # Re-fit

    def remove(self, state, skill_app):
        self.remove_example(state, skill_app) # Remove from X_nom, Y
        if(len(self.X_nom) == 0): return
        nom_ft_weights = self._get_ft_weights()
        self._wdebug(
            f"Calling STAND.fit from remove with X_nom_shape={self.X_nom.shape}, y_len={len(self.Y)}, "
            f"weights_present={nom_ft_weights is not None}"
        )
        self.classifier.fit(self.X_nom, None, self.Y, nom_ft_weights=nom_ft_weights) # Re-fit

    def _predict_with_weighted_features_only(self, X_nom_subset):
        if X_nom_subset.size == 0:
            return 1
        ft_weights = self._get_ft_weights()
        if ft_weights is None:
            return self.classifier.predict(X_nom_subset, None)[0]

        upweighted = np.where(ft_weights > 1.0)[0]
        upweighted = upweighted[upweighted < X_nom_subset.shape[1]]
        n_up = len(upweighted)
        if n_up == 0:
            return 1

        active_up = np.where(X_nom_subset[0, upweighted] != 0)[0]
        n_active = len(active_up)

        if self.weight_debug:
            self._wdebug(
                f"Hint-only predict: upweighted={n_up}, active_upweighted={n_active}"
            )

        return 1 if n_active == n_up else -1

    def predict(self, state, match):   ### if mode is natural langauge, then take self.get_ft_weights and see if all the upweighted feactures are 1 in one hot encoded
        if(len(self.X_nom) == 0): return 1
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)  ### check this against feacture weights
        if self.weighted_hint_only:
            return self._predict_with_weighted_features_only(X_nom_subset)

        if self.weight_debug:
            ft_weights = self._get_ft_weights()
            if ft_weights is not None:
                active = np.where(X_nom_subset[0] != 0)[0]
                active_up = [int(i) for i in active if i < len(ft_weights) and ft_weights[i] > 1.0]
                self._wdebug(
                    f"Predict call active_features={len(active)}, active_upweighted={len(active_up)}"
                )
        prediction = self.classifier.predict(X_nom_subset, None)[0]
        return prediction

    def __str__(self):
        return str(self.classifier)

# --------------------------------------------
# : SklearnDecisionTree

# @register_when(name="decisiontree")
@register_when
class SklearnDecisionTree(BasicSKL):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        from sklearn.tree import DecisionTreeClassifier
        self.classifier = DecisionTreeClassifier()

@register_when
class RandomForest(BasicSKL):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier()

    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)

        # with PrintElapse(f"{type(self).__name__} predict:"):
        probs = self.classifier.predict_proba(X_nom_subset)[0]
        labels = self.classifier.classes_
        best_ind = np.argmax(probs)


        return labels[best_ind] * probs[best_ind]

@register_when
class XGBoost(BasicSKL):
    def __init__(self, skill, **kwargs):
        super().__init__(skill, **kwargs)
        from xgboost import XGBClassifier
        from sklearn.preprocessing import LabelEncoder
        self.classifier = XGBClassifier()
        self.le = LabelEncoder()
        self.le.fit([-1,1])

    def ifit(self, state, skill_app, reward):
        self.add_example(state, skill_app, reward) # Insert into X_nom, Y

        Y = self.le.fit_transform(self.Y)
        with PrintElapse(f"{type(self).__name__} fit:"):
            self.classifier.fit(self.X_nom, Y) # Re-fit


    def predict(self, state, match):
        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)

        # with PrintElapse(f"{type(self).__name__} predict:"):
        probs = self.classifier.predict_proba(X_nom_subset)[0]
        labels = self.classifier.classes_
        labels  = self.le.inverse_transform(labels)
        best_ind = np.argmax(probs)

        return labels[best_ind] * probs[best_ind]

# --------------------------------------------
# : DecisionTree (i.e. STAND library implementation)

@register_when
class DecisionTree(BasicSTAND):
    def __init__(self, skill, impl="decision_tree",
                **kwargs):
        super().__init__(skill, **kwargs)
        from stand.tree_classifier import TreeClassifier
        self.classifier = TreeClassifier(impl, inv_mapper=self.inv_mapper)

# --------------------------------------------
# : STAND

@register_when
class STAND(BasicSTAND):
    def __init__(self, skill, cert_kind="instance_certainty", **kwargs):
        super().__init__(skill, **kwargs)
        from stand.stand import STANDClassifier
        self.classifier = STANDClassifier(inv_mapper=self.inv_mapper, **kwargs)

        if(cert_kind == "instance_certainty"):
            # from stand.stand import instance_certainty
            self.prob_func = STANDClassifier.instance_certainty
        elif(cert_kind == "specific_only"):
            # from stand.stand import instance_certainty
            self.prob_func = STANDClassifier.predict_prob



        
    def predict(self, state, match):
        if(len(self.X_nom) == 0): return 1

        continuous, nominal = self.transform(state, match)
        X_nom_subset = nominal[:self.X_nom.shape[1]].reshape(1,-1)        
        if self.weighted_hint_only:
            return self._predict_with_weighted_features_only(X_nom_subset)

        # prediction = self.classifier.predict(X_nom_subset, None)[0]
        # ia = self.classifier.instance_ambiguity(X_nom_subset[-1], None)
        # print("IA", ia)
        
        # probs, labels  = self.classifier.predict_prob(X_nom_subset, None)
        probs, labels  = self.prob_func(self.classifier, X_nom_subset, None)
        probs = probs[0]
        best_ind = np.argmax(probs)

        return labels[best_ind] * (probs[best_ind])
        # return labels[best_ind] * (probs[best_ind]-probs[~best_ind])
