#include "../basic-abstract-game.h"
#include "../assetgen.h"
#include <set>
#include <queue>
#include <algorithm>

const std::string NAME = "fruitbot";

const float COMPLETION_BONUS = 10.0f;
const int POSITIVE_REWARD = 1;
const int PENALTY = -2.0f; // -4.0f

const int BARRIER = 1;
const int OUT_OF_BOUNDS_WALL = 2;
const int PLAYER_BULLET = 3;
const int BAD_OBJ = 4;
const int GOOD_OBJ = 7;
const int LOCKED_DOOR = 10;
const int LOCK = 11;
const int PRESENT = 12;

const int KEY_DURATION = 8;

const float DOOR_ASPECT_RATIO = 3.25f;

class FruitBotGame : public BasicAbstractGame {
  public:
    float min_dim = 0.0f;
    float bullet_vscale = 0.0f;
    int last_fire_time = 0;

    FruitBotGame()
        : BasicAbstractGame(NAME) {
        mixrate = 0.5f;
        maxspeed = 0.85f;

        min_dim = 5.0f;
        bullet_vscale = 0.5f;
        bg_tile_ratio = -1;

        out_of_bounds_object = OUT_OF_BOUNDS_WALL;
    }

    void load_background_images() override {
        main_bg_images_ptr = &topdown_backgrounds;
    }

    void asset_for_type(int type, std::vector<std::string> &names) override {
        if (type == PLAYER) {
            names.push_back("misc_assets/robot_3Dblue.png");
        } else if (type == BARRIER || type == OUT_OF_BOUNDS_WALL) {
            names.push_back("misc_assets/tileStone_slope.png");
        } else if (type == PLAYER_BULLET) {
            names.push_back("misc_assets/keyRed2.png");
        } else if (type == BAD_OBJ) {
            names.push_back("misc_assets/food1.png");
            names.push_back("misc_assets/food2.png");
            names.push_back("misc_assets/food3.png");
            names.push_back("misc_assets/food4.png");
            names.push_back("misc_assets/food5.png");
            names.push_back("misc_assets/food6.png");
        } else if (type == GOOD_OBJ) {
            names.push_back("misc_assets/fruit1.png");
            names.push_back("misc_assets/fruit2.png");
            names.push_back("misc_assets/fruit3.png");
            names.push_back("misc_assets/fruit4.png");
            names.push_back("misc_assets/fruit5.png");
            names.push_back("misc_assets/fruit6.png");
        } else if (type == LOCKED_DOOR) {
            names.push_back("misc_assets/fenceYellow.png");
        } else if (type == LOCK) {
            names.push_back("misc_assets/lockRed2.png");
        } else if (type == PRESENT) {
            names.push_back("misc_assets/present1.png");
            names.push_back("misc_assets/present2.png");
            names.push_back("misc_assets/present3.png");
        }
    }

    bool will_reflect(int src, int target) override {
        return BasicAbstractGame::will_reflect(src, target) || (src == BAD_OBJ && (target == BARRIER || target == WALL_OBJ));
    }

    bool is_blocked(const std::shared_ptr<Entity> &src, int target, bool is_horizontal) override {
        return BasicAbstractGame::is_blocked(src, target, is_horizontal) || (src->type == PLAYER && target == OUT_OF_BOUNDS_WALL);
    }

    float get_tile_aspect_ratio(const std::shared_ptr<Entity> &ent) override {
        if (ent->type == BARRIER)
            return 1;
        if (ent->type == LOCKED_DOOR)
            return DOOR_ASPECT_RATIO;

        return 0;
    }

    void handle_agent_collision(const std::shared_ptr<Entity> &obj) override {
        BasicAbstractGame::handle_agent_collision(obj);

        // Record collision position and type (normalized 0-1 coordinates)
        step_data.collision_x = obj->x / main_width;
        step_data.collision_y = obj->y / main_height;
        step_data.collision_type = obj->type;

        if (obj->type == BARRIER) {
            step_data.reward += options.fruitbot_reward_wall_hit;
            step_data.done = true;
        } else if (obj->type == BAD_OBJ) {
            step_data.reward += options.fruitbot_reward_negative;
            obj->will_erase = true;
        } else if (obj->type == LOCKED_DOOR) {
            step_data.reward += options.fruitbot_reward_wall_hit;
            step_data.done = true;
        } else if (obj->type == GOOD_OBJ) {
            step_data.reward += options.fruitbot_reward_positive;
            obj->will_erase = true;
        } else if (obj->type == PRESENT) {
            if (!step_data.done) {
            }
            step_data.reward += options.fruitbot_reward_completion;
            step_data.done = true;
            step_data.level_complete = true;
        }
    }

    void handle_collision(const std::shared_ptr<Entity> &src, const std::shared_ptr<Entity> &target) override {
        if (src->type == PLAYER_BULLET) {
            if (target->type == BARRIER) {
                src->will_erase = true;
            } else if (target->type == LOCK) {
                src->will_erase = true;
                target->will_erase = true;

                // find and erase the corresponding door entity
                for (auto ent : entities) {
                    if (ent->type == LOCKED_DOOR && fabs(ent->y - target->y) < 1) {
                        ent->will_erase = true;
                        break;
                    }
                }
            }
        }
    }

    bool use_block_asset(int type) override {
        return BasicAbstractGame::use_block_asset(type) || (type == BARRIER) || (type == LOCKED_DOOR) || (type == PRESENT);
    }

    void choose_center(float &cx, float &cy) override {
        cx = main_width / 2.0;
        cy = agent->y + main_width / 2.0 - 2 * agent->ry;
        visibility = main_width;
    }

    void choose_world_dim() override {
        if (options.distribution_mode == EasyMode) {
            main_width = 10;
        } else {
            main_width = 15;
        }

        main_height = 20;
    }

    void set_action_xy(int move_action) override {
        action_vx = move_action / 3 - 1;
        action_vy = 0.2f; //(move_action % 3) * .2;
        action_vrot = 0;
    }

    void add_walls(float ry, bool use_door, float min_pct) {
        float rw = main_width;
        float wall_ry = 0.3f;
        float lock_rx = 0.25f;
        float lock_ry = 0.45f;

        float pct = min_pct + 0.2f * rand_gen.rand01();

        if (use_door) {
            pct += 0.1f;
            float lock_pct_w = 2 * lock_rx / main_width;
            float door_pct_w = (wall_ry * 2 * DOOR_ASPECT_RATIO) / main_width;
            int num_doors = ceil((pct - 2 * lock_pct_w) / door_pct_w);
            pct = 2 * lock_pct_w + door_pct_w * num_doors;
        }

        float gapw = pct * rw;

        float w1 = rand_gen.rand01() * (rw - gapw);
        float w2 = rw - w1 - gapw;

        add_entity_rxy(w1 / 2, ry, 0, 0, w1 / 2, wall_ry, BARRIER);
        add_entity_rxy(rw - w2 / 2, ry, 0, 0, w2 / 2, wall_ry, BARRIER);

        if (use_door) {
            int is_on_right = rand_gen.randn(2);
            float lock_x = w1 + lock_rx + is_on_right * (gapw - 2 * lock_rx);
            float door_x = w1 + gapw / 2 - (is_on_right * 2 - 1) * lock_rx;

            add_entity_rxy(door_x, ry, 0, 0, gapw / 2 - lock_rx, wall_ry, LOCKED_DOOR);
            add_entity_rxy(lock_x, ry - lock_ry + wall_ry, 0, 0, lock_rx, lock_ry, LOCK);
        }
    }

    void spawn_line_entities(int count, float x_pct, int type, int padding_pct, int object_group_size) {
        if (count <= 0) return;

        float x = std::clamp(x_pct / 100.0f, 0.05f, 0.95f) * main_width;
        float pad = std::clamp(padding_pct / 100.0f, 0.0f, 0.45f) * main_height;
        float y_start = pad + 0.5f;
        float y_end = main_height - pad - 0.5f;
        float span = std::max(0.1f, y_end - y_start);

        for (int i = 0; i < count; i++) {
            float t = (count == 1) ? 0.5f : (float)i / (float)(count - 1);
            float y = y_start + t * span;
            auto ent = add_entity_rxy(x, y, 0, 0, 0.5f, 0.5f, type);
            ent->image_theme = rand_gen.randn(object_group_size);
            fit_aspect_ratio(ent);
        }
    }

    void game_reset() override {
        // Call parent's game_reset FIRST without manipulation
        BasicAbstractGame::game_reset();
        
        last_fire_time = 0;

        int min_sep = 4;
        int num_walls = 10;
        int object_group_size = 6;
        int buf_h = 4;
        float door_prob = 0.125f;
        float min_pct = 0.4f;
        bool force_no_walls = options.fruitbot_force_no_walls;

        if (options.distribution_mode == EasyMode) {
            num_walls = 5;
            object_group_size = options.food_diversity;
            door_prob = 0.0f;
            min_pct = 0.3f;
        }
        
        // Override with custom parameters if provided
        if (options.fruitbot_num_walls >= 0) {
            num_walls = options.fruitbot_num_walls;
        }
        if (options.fruitbot_wall_gap_pct >= 0) {
            min_pct = options.fruitbot_wall_gap_pct / 100.0f;
        }
        if (options.fruitbot_door_prob_pct >= 0) {
            door_prob = options.fruitbot_door_prob_pct / 100.0f;
            // std::cout << "updating door_prob to " << door_prob << std::endl;
        }

        // Clamp gap to avoid degenerate geometry when users pass 100%
        min_pct = std::clamp(min_pct, 0.05f, 0.95f);
        if (options.fruitbot_wall_gap_pct >= 100) {
            force_no_walls = true;
        }
        if (num_walls <= 0) {
            force_no_walls = true;
        }

        // 1. WALLS: Random partition using rand_gen (unless forced off)
        if (!force_no_walls) {
            std::vector<int> partition = rand_gen.partition(std::max(1, main_height - min_sep * num_walls - buf_h), num_walls);
            
            int curr_h = 0;
            for (int part : partition) {
                int dy = min_sep + part;
                curr_h += dy;

                // Random door probability
                bool use_door = (dy > 5) && rand_gen.rand01() < door_prob;
                
                add_walls(curr_h, use_door, min_pct);  // <-- Uses rand_gen for gap positions
            }
        }

        agent->y = agent->ry;

        // Handle range=0 case to avoid modulo by zero in randn()
        int num_good = (options.fruitbot_num_good_range > 0 ? rand_gen.randn(options.fruitbot_num_good_range) : 0) + options.fruitbot_num_good_min;
        int num_bad = (options.fruitbot_num_bad_range > 0 ? rand_gen.randn(options.fruitbot_num_bad_range) : 0) + options.fruitbot_num_bad_min;

        for (int i = 0; i < main_width; i++) {
            auto present = add_entity_rxy(i + 0.5f, main_height - 0.5f, 0, 0, 0.5f, 0.5f, PRESENT);
            choose_random_theme(present);
        }

        bool use_line_layout = options.fruitbot_layout_mode == 1;

        // 3. FOOD: structured layout or random spawn
        if (use_line_layout) {
            spawn_line_entities(num_good, options.fruitbot_good_line_x_pct, GOOD_OBJ, options.fruitbot_line_padding_pct, object_group_size);
            spawn_line_entities(num_bad, options.fruitbot_bad_line_x_pct, BAD_OBJ, options.fruitbot_line_padding_pct, object_group_size);
        } else {
            if (num_good > 0) {
                spawn_entities(num_good, 0.5f, GOOD_OBJ, 0, 0, main_width, main_height);
            }
            if (num_bad > 0) {
                spawn_entities(num_bad, 0.5f, BAD_OBJ, 0, 0, main_width, main_height);
            }

            // 4. FOOD SPRITES: Random sprite selection
            for (auto ent : entities) {
                if (ent->type == GOOD_OBJ || ent->type == BAD_OBJ) {
                    ent->image_theme = rand_gen.randn(object_group_size);
                    fit_aspect_ratio(ent);
                }
            }
        }
        
        agent->rotation = -1.0f * PI / 2.0f;
    }

    void game_step() override {
        BasicAbstractGame::game_step();
        
        // Small reward for each step survived (encourages forward progress)
        if (options.fruitbot_reward_step != 0.0f) {
            step_data.reward += options.fruitbot_reward_step;
        }

        if (special_action == 1 && (cur_time - last_fire_time) >= KEY_DURATION) {
            float vx = 0.0f;
            float vy = 1.0f;
            auto new_bullet = add_entity(agent->x, agent->y, vx * bullet_vscale, vy * bullet_vscale, 0.25f, PLAYER_BULLET);
            new_bullet->expire_time = KEY_DURATION;
            new_bullet->collides_with_entities = true;
            last_fire_time = cur_time;
        }
    }

    void serialize(WriteBuffer *b) override {
        BasicAbstractGame::serialize(b);
        b->write_float(min_dim);
        b->write_float(bullet_vscale);
        b->write_int(last_fire_time);
    }

    void deserialize(ReadBuffer *b) override {
        BasicAbstractGame::deserialize(b);
        min_dim = b->read_float();
        bullet_vscale = b->read_float();
        last_fire_time = b->read_int();
    }
};

REGISTER_GAME(NAME, FruitBotGame);
